# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import subprocess
import sys
import time
from dataclasses import replace

import psutil
import pytest

from tools.tokencake_experiments.campaign import DEVICES, ROOT, SOURCE, Case
from tools.tokencake_experiments.driver import Runner, audit
from tools.tokencake_experiments.materialize import digest, materialize
from tools.tokencake_experiments.preflight import (
    adapter,
    carry_exclusions,
    inherited_environment,
)
from tools.tokencake_experiments.report import Identity, content_hash, measurements
from tools.tokencake_experiments.runtime import (
    Activity,
    Monitor,
    Process,
    cpu_set,
    write_json,
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


def test_planning_cli_exact_fifteen_cases_and_no_truncation(tmp_path):
    destination = tmp_path / "plan.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.tokencake_experiments.driver",
            "plan",
            "--output",
            str(destination),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    plan = json.loads(destination.read_text())
    queues = plan["queues"]
    assert [(case["mode"], case["qps"]) for case in queues["primary"][0]] == [
        (mode, qps) for mode in ("native", "offload-agent") for qps in (1, 0.5, 0.1)
    ]
    assert [case["mode"] for case in queues["primary"][1]] == ["agent"] * 3
    assert [case["mode"] for case in queues["references"][0]] == [
        "old-offload-agent"
    ] * 3
    assert [case["mode"] for case in queues["references"][1]] == ["mooncake"] * 3
    cases = [case for stage in queues.values() for queue in stage for case in queue]
    assert len(cases) == plan["initial_case_count"] == 15
    for case in cases:
        server, client = case["server_command"], case["client_command"]
        assert server[:3] == client[:3] == ["taskset", "-c", case["device"]["cpus"]]
        assert client[client.index("--num_requests") + 1] == "24"
        assert not any(
            "smoke" in item or "tokens_cap" in item or "--debug" in item
            for item in client
        )
        if case["mode"] == "native":
            assert (
                "--additional-config" not in server
                and "--kv-offloading-size" not in server
            )
            assert client[-2:] == ["--tokencake-mode", "native"]
        elif case["mode"] == "agent":
            assert json.loads(server[server.index("--additional-config") + 1]) == {
                "tokencake": {"offload": {"enabled": False}}
            }
            assert "--kv-offloading-size" not in server
        elif case["mode"] == "offload-agent":
            assert server[server.index("--kv-offloading-size") + 1] == "100"
        elif case["mode"] == "old-offload-agent":
            assert str(SOURCE / "vllm_serving.py") in client
            assert (
                "--tokencake-mode" not in client
                and "--disable_mcp_notifications" not in client
            )
        else:
            assert client[-1] == "--disable_mcp_notifications"
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


def test_completed_case_is_auditable_with_unchanged_source_analyzer(tmp_path):
    if not SOURCE.exists():
        pytest.skip("Requires the frozen source checkout")
    checkout = tmp_path / "launcher"
    materialize(SOURCE, checkout, patched=False)
    case_dir = tmp_path / "case"
    contracts = [{"input": index} for index in range(24)]
    apps = {
        str(index): {
            "app_finished": True,
            "arrival_offset_s": index,
            "app_latency": 1 + index,
            "expected_node_names": ["local", "generation"],
            "frozen_workload_contract": contracts[index],
            "request_info": {
                "local": {"execution_kind": "local", "finish_reason": "local"},
                "generation": {
                    "execution_kind": "llm",
                    "finish_reason": "length",
                    "generated_tokens": 500,
                    "prompt_tokens": 100,
                    "processed_tokens": 600,
                },
            },
        }
        for index in range(24)
    }
    write_json(case_dir / "app_results/completed.json", apps)
    (case_dir / "server.log").write_text("")
    (case_dir / "client.log").write_text("[debug][a] failed, retrying... (1/3)\n")
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
    assert source["exclusion_reasons"] == ["mooncake-no-store-operations"]
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
    assert result["correctness"]["token_counts"]["generated"] == 12000
    assert result["native_mooncake_telemetry"]["successful_put_calls"] == 12
    assert result["attempt_diagnostics"]["prompt_halvings"] == 1
    for path, expected in result["artifacts"].items():
        assert digest(case_dir / path) == expected
    apps["0"]["arrival_offset_s"] = 1
    (case_dir / "app_results/completed.json").write_text(json.dumps(apps))
    rejected = audit(case_dir, source, execution, frozen)
    assert "arrival_schedule_mismatch" in rejected["exclusion_reasons"]
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
        assert not rejected["qualifying"]
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
    while child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
        assert time.monotonic() < deadline
        time.sleep(0.01)


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
        lambda kind, fields: processes
        if kind == "compute-apps"
        else [{"temperature.gpu": "99"}],
    )
    monitor = Monitor(DEVICES[0], activity, tmp_path / "monitor.jsonl")
    sample = monitor.sample()
    assert sample["activity"][DEVICES[1].uuid]["case"] == "agent"
    assert not monitor.summary["external_gpu_process_detected"]
    assert not monitor.summary["thermal_validity_enforced"]
    processes.append(foreign)
    assert monitor.sample()["foreign"] == [foreign]
    assert monitor.summary["external_gpu_process_detected"]
