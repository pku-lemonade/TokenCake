# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from tools.tokencake_experiments import driver, preflight, provenance
from tools.tokencake_experiments.agent_bench import inputs, service
from tools.tokencake_experiments.agent_bench.experiment import tasks_for
from tools.tokencake_experiments.analysis import (
    summarize_attempt,
    validate_dag_completion,
)
from tools.tokencake_experiments.campaign import (
    PACKAGE,
    ROOT,
    Case,
    Settings,
    client_command,
)
from tools.tokencake_experiments.dataset import load_dataset
from tools.tokencake_experiments.materialize import digest
from tools.tokencake_experiments.runtime import write_json


def test_relative_interpreter_retains_venv_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert Settings(python=".venv/bin/python").python == str(
        tmp_path / ".venv/bin/python"
    )


def test_prepare_and_run_do_not_require_environment_checks(tmp_path, monkeypatch):
    dataset = load_dataset(PACKAGE / "datasets/conversation-tools.json")
    payload = dataset.model_dump()
    payload.update(
        name="custom", profile="custom", applications=payload["applications"][:1]
    )
    payload["provenance"]["seed"] = 7
    dataset_path = tmp_path / "custom.json"
    write_json(dataset_path, payload)
    monkeypatch.setattr(preflight, "gpu_devices", lambda: [])

    def no_external_commands(*args, **kwargs):
        raise FileNotFoundError("No git, uv, nvidia-smi or native runtime available")

    monkeypatch.setattr(preflight.subprocess, "run", no_external_commands)
    output = tmp_path / "run"
    case = Case("native", 0.3, gpu_index=7)
    settings = Settings(model="org/uncached-model", gpus=(7,))
    config = preflight.prepare(
        output, settings=settings, cases=[case], workload_dataset=dataset_path
    )
    assert config["code"]["baseline"] == config["code"]["target"]
    assert config["settings"]["model"] == "org/uncached-model"
    assert config["parameters"]["num_requests"] == 1
    assert config["parameters"]["task"] == "custom"
    assert config["parameters"]["seed"] == 7
    assert config["mooncake"] is None
    assert not (output / "environment-target.json").exists()
    assert digest(output / "launcher/analysis.py") == digest(PACKAGE / "analysis.py")
    command, _, _ = client_command(
        case,
        1,
        output,
        output / "launcher",
        output / "arrivals-0.3.json",
        settings=settings,
        workload=config["parameters"],
    )
    assert command[command.index("--num_requests") + 1] == "1"
    assert command[command.index("--task") + 1] == "custom"
    (output / "launcher/analysis.py").write_text("# changed after preparation")
    runner = driver.Runner(output)
    launched = []
    monkeypatch.setattr(
        runner, "run_queues", lambda queues, **kw: launched.extend(queues)
    )
    monkeypatch.setattr(runner, "report", lambda **kw: {})
    runner.run()
    assert launched == [[case]]


def test_small_agent_dataset_needs_no_environment_manifest(tmp_path, monkeypatch):
    data = (
        tmp_path
        / "sources/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"
        / "BFCL_v4_multi_turn_base.json"
    )
    data.parent.mkdir(parents=True)
    data.write_text('{"id": "custom_0", "question": []}\n')

    def no_git(*args):
        raise FileNotFoundError("git unavailable")

    monkeypatch.setattr(provenance, "git", no_git)
    manifest = inputs.capture(
        tmp_path,
        tmp_path / "run",
        seed="custom",
        workers=1,
        status="candidate",
        settings=Settings(model="org/uncached-model", gpus=(7,)),
    )
    assert manifest["model_path"] == "org/uncached-model"
    assert manifest["target"]["commit"] is None
    assert manifest["swe"]["ids"] == []
    assert list(manifest["bfcl"]) == ["multi_turn_base"]
    assert tasks_for(manifest, "bfcl", "full") == [{"id": "custom_0", "question": []}]
    data.write_text('{"id": "custom_1", "question": []}\n')
    assert tasks_for(manifest, "bfcl", "full") == [{"id": "custom_1", "question": []}]


def test_unreadable_cgroup_information_is_optional(monkeypatch):
    def unreadable(*args, **kwargs):
        raise PermissionError("cgroup statistics unavailable")

    monkeypatch.setattr(Path, "read_text", unreadable)
    assert inputs.resource_limit("memory.max") is None


@pytest.mark.parametrize("grading", [False, True])
def test_agent_workers_can_use_current_interpreter(tmp_path, grading):
    assert service.worker_python(tmp_path, grading=grading) == Settings.python


@pytest.fixture
def completed_dag():
    dataset = load_dataset(PACKAGE / "datasets/conversation-tools.json")
    dataset = dataset.model_copy(update={"applications": dataset.applications[:1]})
    apps = {
        "0": {
            "app_finished": True,
            "app_latency": 2.0,
            "frozen_workload_contract": dataset.contract(dataset.applications[0]),
            "request_info": {
                node.name: {
                    "execution_kind": "llm" if node.kind == "llm" else "local",
                    "finish_reason": "length" if node.kind == "llm" else "local",
                    "generated_tokens": node.max_tokens,
                }
                for node in dataset.nodes
            },
        }
    }
    return dataset, apps


@pytest.mark.parametrize(
    "change,reason",
    [
        ("missing_node", "dag_node_mismatch"),
        ("short_generation", "generation_budget_mismatch"),
        ("failed_node", "invalid_terminal_status"),
        ("contract", "workload_contract_mismatch"),
        ("unfinished", "incomplete_application"),
        ("server", "server_error"),
    ],
)
def test_local_analysis_rejects_incomplete_work(completed_dag, change, reason):
    dataset, apps = completed_dag
    valid = validate_dag_completion(apps, "", 1, dataset=dataset)
    assert valid["passed"]
    assert valid["observed_workload_hash"] == dataset.freeze([1.0])["workload_sha256"]
    app = apps["0"]
    node = next(node for node in dataset.nodes if node.kind == "llm")
    record = app["request_info"][node.name]
    log = ""
    if change == "missing_node":
        del app["request_info"][node.name]
    elif change == "short_generation":
        record["generated_tokens"] -= 1
    elif change == "failed_node":
        record["finish_reason"] = "error"
    elif change == "contract":
        app["frozen_workload_contract"]["task"] = "different"
    elif change == "unfinished":
        app["app_finished"] = False
    else:
        log = "CUDA out of memory"
    result = validate_dag_completion(apps, log, 1, dataset=dataset)
    assert not result["passed"]
    assert reason in result["exclusion_reasons"]


def test_local_summary_excludes_retried_work(completed_dag, tmp_path):
    dataset, apps = completed_dag
    write_json(tmp_path / "app_results/apps.json", apps)
    (tmp_path / "server.log").write_text("")
    client = tmp_path / "client.log"
    client.write_text("")

    def analyze():
        return summarize_attempt(
            tmp_path,
            {"point": {"num_requests": 1}},
            3.0,
            {},
            {"passed": True},
            dataset=dataset,
        )

    result = analyze()
    assert result["exclusion_reasons"] == []
    assert result["performance"]["average_app_latency_s"] == 2.0
    assert result["performance"]["throughput_rps"] == pytest.approx(1 / 3)
    client.write_text("request failed, retrying... (1/3)")
    assert "request_retries" in analyze()["exclusion_reasons"]


@pytest.mark.parametrize("mode", ["base", "agent", "offload", "agent_offload"])
@pytest.mark.parametrize("gpu_discovery", [True, False])
def test_agent_services_use_selected_model_gpu_and_current_checkout(
    tmp_path, monkeypatch, mode, gpu_discovery
):
    launched = []

    class Process:
        process = SimpleNamespace(pid=123, poll=lambda: None, returncode=0)

        def __init__(self, command, environment, cwd, log, *, cpus):
            launched.append((command, environment, cwd, cpus))

        def close(self):
            pass

    class Monitor:
        summary = {}

        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    client = httpx.Client
    monkeypatch.setattr(
        service.httpx,
        "Client",
        lambda **kwargs: client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    404 if request.url.path == "/metrics" else 200, json={}
                )
            )
        ),
    )
    monkeypatch.setattr(service, "free_port", lambda p: p)
    monkeypatch.setattr(service, "Process", Process)
    monkeypatch.setattr(service, "MetricsMonitor", Monitor)
    monkeypatch.setattr(service, "PressureMonitor", Monitor)
    monkeypatch.setattr(
        service,
        "gpu_devices",
        lambda: ([{"index": "7", "uuid": "GPU-current"}] if gpu_discovery else []),
    )
    with service.Service(
        tmp_path,
        tmp_path / "server",
        mode,
        gpu=7,
        model="/models/selected",
        settings={
            "python": "/env/.venv/bin/python",
            "cpu_kv_gib": 2,
            "max_model_len": 8192,
            "dtype": "float16",
            "gpu_memory_utilization": 0.7,
        },
    ):
        command, environment, cwd, cpus = launched[0]
        assert command[0] == "/env/.venv/bin/python"
        assert command[command.index("serve") + 1] == "/models/selected"
        assert command[command.index("--max-model-len") + 1] == "8192"
        assert command[command.index("--dtype") + 1] == "float16"
        assert environment["CUDA_VISIBLE_DEVICES"] == (
            "GPU-current" if gpu_discovery else "7"
        )
        assert environment["PYTHONPATH"] == str(ROOT)
        assert cwd == ROOT and cpus is None and "taskset" not in command
        assert ("--additional-config" in command) == (mode != "base")
        if mode in ("offload", "agent_offload"):
            assert command[command.index("--kv-offloading-size") + 1] == "2"
    recorded = json.loads((tmp_path / "server/launch.json").read_text())
    assert recorded["selected_gpu_index"] == 7
    assert (tmp_path / "server/metrics-start.error.txt").is_file()
