# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import replace

import httpx
import psutil

from tools.tokencake_experiments.campaign import (
    ROOT,
    Case,
    Device,
    Settings,
    client_command,
    server_command,
)
from tools.tokencake_experiments.dataset import load_dataset
from tools.tokencake_experiments.driver import Runner, audit, plan
from tools.tokencake_experiments.materialize import digest, materialize
from tools.tokencake_experiments.preflight import (
    adapter,
    carry_exclusions,
    hardware_info,
    inherited_environment,
)
from tools.tokencake_experiments.report import Identity, content_hash, measurements
from tools.tokencake_experiments.runtime import (
    Activity,
    MetricsMonitor,
    Monitor,
    Process,
    cpu_set,
    write_json,
)

DEVICES = (Device(2, "GPU-test-a", cpus="0-3"), Device(5, "GPU-test-b", cpus="4-7"))


def test_diagnostic_case_freezes_runtime_and_keeps_full_workload(tmp_path):
    settings = Settings(model="/models/local", gpus=(2,), cpus="0-3")
    case = Case("agent", 1.0, gpu_index=2)
    snapshot = tmp_path / "runtime"
    server, environment, cwd = server_command(
        case, 8055, target_checkout=snapshot, settings=settings
    )
    client, _, _ = client_command(
        case,
        8055,
        tmp_path / "case",
        tmp_path / "launcher",
        tmp_path / "arrivals.json",
        settings=settings,
    )
    assert environment["PYTHONPATH"] == str(snapshot)
    assert environment["CUDA_VISIBLE_DEVICES"] == "2"
    assert cwd == snapshot
    assert server[:3] == client[:3] == ["taskset", "-c", "0-3"]
    assert client[client.index("--num_requests") + 1] == "24"
    assert server[server.index("serve") + 1] == "/models/local"
    assert (
        server_command(Case("native", 1.0), 8055, target_checkout=snapshot)[2]
        == snapshot
    )


def test_environment_fingerprint_keeps_performance_knobs(monkeypatch):
    monkeypatch.setenv("VLLM_MAX_NUM_BATCHED_TOKENS", "2048")
    monkeypatch.setenv("TOKENCAKE_POLICY", "test")
    monkeypatch.setenv("VLLM_API_KEY", "test-secret")
    monkeypatch.setenv("HF_TOKEN", "test-secret")
    environment = inherited_environment()
    assert environment["VLLM_MAX_NUM_BATCHED_TOKENS"] == "2048"
    assert environment["TOKENCAKE_POLICY"] == "test"
    assert "VLLM_API_KEY" not in environment and "HF_TOKEN" not in environment


def test_component_matrix_has_twenty_distinct_cases_and_independent_switches():
    matrix = plan(components=True)
    cases = [case for queue in matrix["queues"]["components"] for case in queue]
    assert len(cases) == matrix["initial_case_count"] == 20
    assert len({case["name"] for case in cases}) == 20
    assert {(case["mode"], case["qps"]) for case in cases} == {
        (mode, qps)
        for mode in ("native", "agent", "offload", "offload-agent")
        for qps in (1.0, 0.5, 0.2, 0.1, 0.05)
    }
    assert Case("native", 0.05).name != Case("native", 0.1).name
    assert Case("native", 1).name.endswith("qps-1.0")
    for case in cases:
        command = case["server_command"]
        mode = case["mode"]
        if mode == "native":
            assert "--additional-config" not in command
        else:
            config = json.loads(command[command.index("--additional-config") + 1])[
                "tokencake"
            ]
            assert config.get("scheduling", {}).get("enabled", True) == (
                mode != "offload"
            )
            assert config.get("offload", {}).get("enabled", True) == (mode != "agent")
        assert ("--kv-offloading-size" in command) == (
            mode in ("offload", "offload-agent")
        )
        assert case["client_command"][-2:] == ["--tokencake-mode", mode]


def test_metrics_sampling_retains_occupancy_labels_and_transfer_totals(tmp_path):
    metrics = (
        "vllm:kv_cache_usage_perc 0.75\n"
        'vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"} 1024\n'
        'vllm:latency_bucket{le="1"} 3\n'
        "vllm:latency_created 100\n"
        "vllm:latency_sum 2\n"
        "vllm:latency_count 3\n"
    )
    monitor = MetricsMonitor(1, tmp_path / "samples.jsonl")
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=metrics))
    ) as client:
        sample = monitor.sample(client)
    observed = {item["name"]: item for item in sample["metrics"]}
    assert observed["vllm:kv_cache_usage_perc"]["value"] == 0.75
    assert observed["vllm:kv_offload_total_bytes_total"]["labels"] == {
        "transfer_type": "CPU_to_GPU"
    }
    assert (
        "vllm:latency_bucket" not in observed and "vllm:latency_created" not in observed
    )
    assert observed["vllm:latency_count"]["value"] == 3


def test_planning_cli_uses_current_checkout_on_arbitrary_single_gpu(tmp_path):
    destination = tmp_path / "plan.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.tokencake_experiments.driver",
            "plan",
            "--model",
            "/models/custom",
            "--gpus",
            "5",
            "--output",
            str(destination),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    plan = json.loads(destination.read_text())
    queues = plan["queues"]["primary"]
    assert len(queues) == 1
    assert len(queues[0]) == plan["initial_case_count"] == 9
    assert {case["mode"] for case in queues[0]} == {"native", "agent", "offload-agent"}
    for case in queues[0]:
        server, client = case["server_command"], case["client_command"]
        assert "taskset" not in server and "taskset" not in client
        assert case["device"]["index"] == 5
        assert case["server_cwd"] == str(ROOT)
        assert server[server.index("serve") + 1] == "/models/custom"
        assert client[client.index("--num_requests") + 1] == "24"
        assert case["server_environment"]["CUDA_VISIBLE_DEVICES"] == "5"
        if case["mode"] == "native":
            assert (
                "--additional-config" not in server
                and "--kv-offloading-size" not in server
            )
    rejected = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.tokencake_experiments.driver",
            "run",
            str(tmp_path),
            "--smoke_max_finished_nodes",
            "1",
        ],
        cwd=ROOT,
        capture_output=True,
    )
    assert rejected.returncode == 2


def test_preflight_selects_actual_devices_without_binding_other_gpus(monkeypatch):
    monkeypatch.setattr(
        "tools.tokencake_experiments.preflight.gpu_devices",
        lambda: [
            {
                "index": "2",
                "uuid": "GPU-new",
                "name": "NVIDIA H100",
                "memory.total": "80000",
            },
            {
                "index": "0",
                "uuid": "GPU-busy",
                "name": "other",
                "memory.total": "24000",
            },
        ],
    )
    result = hardware_info(Settings(gpus=(2,)))
    assert result["devices"][0]["uuid"] == "GPU-new"
    # Unavailable devices are left to the actual serving command.
    result = hardware_info(Settings(gpus=(3,)))
    assert result["devices"][0]["index"] == 3
    assert result["devices"][0]["uuid"] == ""


def test_completed_case_is_auditable_with_local_analyzer(tmp_path):
    checkout = tmp_path / "launcher"
    materialize(ROOT, checkout)
    case_dir = tmp_path / "case"
    dataset = load_dataset(checkout / "workload-dataset.json")
    contracts = [dataset.contract(app) for app in dataset.applications]
    apps = {
        app.id: {
            "app_finished": True,
            "arrival_offset_s": index,
            "app_latency": 1 + index,
            "expected_node_names": [node.name for node in dataset.nodes],
            "frozen_workload_contract": contracts[index],
            "request_info": {
                node.name: {
                    "execution_kind": "llm" if node.kind == "llm" else "local",
                    "finish_reason": "length" if node.kind == "llm" else "local",
                    "generated_tokens": node.max_tokens,
                    "prompt_tokens": 100 if node.kind == "llm" else 0,
                    "processed_tokens": node.max_tokens
                    + (100 if node.kind == "llm" else 0),
                }
                for node in dataset.nodes
            },
        }
        for index, app in enumerate(dataset.applications)
    }
    write_json(case_dir / "app_results/completed.json", apps)
    (case_dir / "server.log").write_text("")
    (case_dir / "client.log").write_text("")
    samples = (
        'vllm:mooncake_store_operation_total{operation="save_put",status="ok"} 12\n'
    )
    (case_dir / "metrics.prom").write_text(samples)
    contamination = {
        "external_gpu_process_detected": False,
        "monitor_query_failed": False,
        "affinity_violation": False,
        "thermal_validity_enforced": False,
        "samples": 2,
    }
    parameters = {
        "case_dir": str(case_dir),
        "attempt": {"point": {"num_requests": 24}, "mode_config": {"kind": "mooncake"}},
        "total_e2e_s": 100,
        "contamination": contamination,
        "state_isolation": {"passed": True},
    }
    adapter("analyze", checkout, parameters, case_dir / "source-analysis.json")
    source = json.loads((case_dir / "source-analysis.json").read_text())
    assert source["exclusion_reasons"] == []
    identity = Identity(
        Case("mooncake", 1), "code", "env", "launcher", "input", "config"
    )
    execution = {
        "identity": identity.payload(),
        "launch": 0,
        "client_exit_code": 0,
        "server_alive": True,
        "contamination": contamination,
        "performance": {"total_e2e_s": 100},
    }
    frozen = {
        "workload_sha256": content_hash(contracts),
        "arrivals": {"1.0": list(range(24))},
    }
    result = audit(case_dir, source, execution, frozen)
    assert result["qualifying"]
    assert result["performance"]["p50_app_latency_s"] == 12.5
    assert result["correctness"]["application_count"] == 24
    assert (
        result["correctness"]["token_counts"]["generated"]
        == sum(node.max_tokens for node in dataset.nodes) * 24
    )
    assert result["native_mooncake_telemetry"]["successful_put_calls"] == 12
    assert result["attempt_diagnostics"]["prompt_halvings"] == 0
    for path, expected in result["artifacts"].items():
        assert digest(case_dir / path) == expected
    apps["0"]["arrival_offset_s"] = 1
    (case_dir / "app_results/completed.json").write_text(json.dumps(apps))
    rejected = audit(case_dir, source, execution, frozen)
    assert rejected["qualifying"]
    apps["0"]["arrival_offset_s"] = 0
    (case_dir / "app_results/completed.json").write_text(json.dumps(apps))
    for flag in (
        "affinity_violation",
        "monitor_query_failed",
        "external_gpu_process_detected",
    ):
        rejected = audit(
            case_dir,
            source,
            execution | {"contamination": contamination | {flag: True}},
            frozen,
        )
        assert rejected["qualifying"]
    (case_dir / "metrics.prom").write_text(
        samples
        + 'vllm:mooncake_store_operation_failed_keys_total{operation="save_put"} 1\n'
    )
    assert not audit(case_dir, source, execution, frozen)["qualifying"]


def test_interrupted_launch_is_persisted_and_counts_toward_budget(tmp_path):
    identity = Identity(Case("native", 1), "code", "env", "launcher", "input", "config")
    write_json(tmp_path / "frozen.json", {"identities": [identity.payload()]})
    path = tmp_path / "cases" / identity.case.name / "launch-0/attempt.json"
    write_json(path, {"identity": identity.payload(), "launch": 0})
    runner = Runner(tmp_path)
    rows = measurements(identity, runner.results)
    assert len(rows) == 1 and not rows[0]["qualifying"]
    assert rows[0]["exclusion_reasons"] == ["interrupted_attempt"]
    assert path.with_name("result.json").exists()
    assert Runner(tmp_path).results == runner.results


def test_recovery_keeps_first_qps_in_initial_queue(tmp_path, monkeypatch):
    previous = tmp_path / "previous"
    current = tmp_path / "current"
    identity = Identity(
        Case("native", 1.0), "code", "env", "launcher", "input", "config"
    )
    original = identity.payload()
    case_dir = previous / "cases" / identity.case.name / "launch-0"
    write_json(case_dir / "attempt.json", {"identity": original, "launch": 0})
    write_json(
        case_dir / "result.json",
        {
            "identity": original,
            "launch": 0,
            "qualifying": False,
            "result_path": str(case_dir / "result.json"),
        },
    )
    identities = [
        replace(identity, case=Case("native", qps)) for qps in (1.0, 0.5, 0.1)
    ]
    exclusions = carry_exclusions(previous, identities)
    assert exclusions[0]["identity"] == original
    write_json(
        current / "frozen.json", {"identities": [case.payload() for case in identities]}
    )
    write_json(current / "prior-exclusions.json", exclusions)
    runner = Runner(current)
    launched = []
    monkeypatch.setattr(
        runner,
        "run_case",
        lambda case: launched.append(
            (case.case.qps, len(measurements(case, runner.results)))
        ),
    )
    runner.run_queues([[case.case for case in identities]], initial=True)
    assert launched == [(1.0, 1), (0.5, 0), (0.1, 0)]


def test_process_affinity_and_child_cleanup(tmp_path):
    cpus = str(min(psutil.Process().cpu_affinity()))
    code = (
        "import subprocess, sys, time; "
        "p = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(300)']); "
        "print(p.pid, flush=True); time.sleep(300)"
    )
    log = tmp_path / "process.log"
    process = Process([sys.executable, "-c", code], {}, ROOT, log, cpus=cpus)
    try:
        deadline = time.monotonic() + 10
        while not log.read_text().strip():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        child = psutil.Process(int(log.read_text().strip()))
        assert psutil.Process(process.process.pid).cpu_affinity() == [int(cpus)]
        assert child.cpu_affinity() == [int(cpus)]
    finally:
        process.close()
    assert process.process.poll() is not None
    deadline = time.monotonic() + 5
    while True:
        try:
            if not child.is_running() or child.status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        assert time.monotonic() < deadline
        time.sleep(0.01)


def test_offload_modes_can_run_on_different_gpus_concurrently(tmp_path, monkeypatch):
    cases = [Case("offload", 1.0, gpu_index=2), Case("offload-agent", 1.0, gpu_index=5)]
    identities = [
        Identity(case, "code", "env", "launcher", "input", "config") for case in cases
    ]
    write_json(
        tmp_path / "frozen.json",
        {"identities": [item.payload() for item in identities]},
    )
    runner = Runner(tmp_path)
    both_started = threading.Barrier(2)
    completed = []

    def run_case(identity):
        both_started.wait(timeout=5)
        completed.append(identity.case)

    monkeypatch.setattr(runner, "run_case", run_case)
    runner.run_queues([[cases[0]], [cases[1]]], initial=True)
    assert set(completed) == set(cases)


def test_gpu_monitor_records_peer_without_thermal_invalidation(tmp_path, monkeypatch):
    activity = Activity()
    activity.set(DEVICES[0], "native", [os.getpid()])
    activity.set(DEVICES[1], "agent", [12345])
    monkeypatch.setattr(
        psutil.Process, "cpu_affinity", lambda self: sorted(cpu_set(DEVICES[0].cpus))
    )
    foreign = {
        "gpu_uuid": DEVICES[0].uuid,
        "pid": "999999",
        "process_name": "outside",
        "used_gpu_memory": "1",
    }
    controlled = {
        "gpu_uuid": DEVICES[0].uuid,
        "pid": str(os.getpid()),
        "process_name": "case",
        "used_gpu_memory": "1",
    }
    peer = {
        "gpu_uuid": DEVICES[1].uuid,
        "pid": "12345",
        "process_name": "peer",
        "used_gpu_memory": "1",
    }
    processes = [controlled, peer]
    monkeypatch.setattr(
        "tools.tokencake_experiments.runtime.query_nvidia",
        lambda kind, fields: (
            processes if kind == "compute-apps" else [{"temperature.gpu": "99"}]
        ),
    )
    monitor = Monitor(DEVICES[0], activity, tmp_path / "monitor.jsonl")
    sample = monitor.sample()
    assert sample["activity"][DEVICES[1].uuid]["case"] == "agent"
    assert not monitor.summary["external_gpu_process_detected"]
    assert not monitor.summary["thermal_validity_enforced"]
    processes.append(foreign)
    assert monitor.sample()["foreign"] == [foreign]
    assert monitor.summary["external_gpu_process_detected"]
