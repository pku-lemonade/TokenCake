# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed input validation, complete DAG execution, and snapshot integrity."""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from copy import deepcopy

import httpx
import pytest

from tools.tokencake_experiments import dataset_client
from tools.tokencake_experiments.campaign import PACKAGE, SOURCE, Case, client_command
from tools.tokencake_experiments.component_graph import export_graph
from tools.tokencake_experiments.dataset import (
    Dataset,
    compose_input,
    load_dataset,
    text_hash,
)
from tools.tokencake_experiments.materialize import (
    DATASET_NAME,
    WORKLOAD_PROFILES,
    digest,
    git,
    materialize,
)
from tools.tokencake_experiments.preflight import adapter


def load_profile(profile="conversation-tools"):
    return load_dataset(PACKAGE / "datasets" / f"{profile}.json")


@pytest.mark.parametrize("profile", WORKLOAD_PROFILES)
def test_fixed_profiles_preserve_full_workload(profile):
    dataset = load_profile(profile)
    assert len(dataset.applications) == 24
    assert len(dataset.nodes) == 29
    assert sum(len(node.predecessors) for node in dataset.nodes) == 36
    assert sum(node.kind == "llm" for node in dataset.nodes) == 27
    assert sum(node.max_tokens for node in dataset.nodes) == (
        6464 if profile == "conversation-tools" else 11300
    )
    by_name = {node.name: node for node in dataset.nodes}
    assert by_name["programmer_2_validate_patch"].tool.duration_s == 8
    assert by_name["reviser_2_validate_patch"].tool.duration_s == 10
    assert len(by_name["reviewer_1"].predecessors) == 3
    assert [app.initial_input for app in dataset.applications] == [
        app.initial_input for app in load_profile("frozen").applications
    ]
    frozen = dataset.freeze([1.0, 0.5, 0.1])
    assert frozen["arrivals"]["0.5"] == [index * 2 for index in range(24)]
    if profile == "conversation-tools":
        # Contract from the last accepted campaign, before the JSON migration.
        assert (
            frozen["workload_sha256"]
            == "e2de7eb7d6187bc0653c3d8967f769d38aa78da7d47043f0e271711ab9b1e65c"
        )


@pytest.mark.parametrize("profile", WORKLOAD_PROFILES)
def test_every_application_uses_live_outputs_and_identical_inputs_across_modes(
    profile, monkeypatch
):
    dataset = load_profile(profile)
    events = []

    async def event(_, payload, **kwargs):
        events.append(payload)

    async def no_sleep(_):
        await asyncio.sleep(0)

    def respond(request):
        body = json.loads(request.content)
        generated = "\nLIVE_MODEL_OUTPUT_" + text_hash(body["prompt"]) + "\n"
        return httpx.Response(
            200,
            json={
                "id": body["request_id"],
                "choices": [{"text": generated, "finish_reason": "length"}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": body["max_tokens"],
                    "total_tokens": 100 + body["max_tokens"],
                },
            },
        )

    monkeypatch.setattr(dataset_client, "post_event", event)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            for app in dataset.applications:
                fingerprints = []
                for mode in ("native", "agent", "offload", "offload-agent"):
                    events.clear()
                    info, io = await dataset_client.run_application(
                        dataset,
                        app,
                        client=client,
                        base_url="http://test/v1",
                        model="test",
                        mode=mode,
                        origin=time.time(),
                        sleep=no_sleep,
                    )
                    assert (
                        info["app_finished"] and info["application_internal_finished"]
                    )
                    assert len(info["request_info"]) == 29 and len(io) == 27
                    assert sum(row["generated_tokens"] for row in io.values()) == sum(
                        node.max_tokens for node in dataset.nodes
                    )
                    assert len(events) == (
                        2 * sum(node.tool is not None for node in dataset.nodes)
                        if mode in ("offload", "offload-agent")
                        else 0
                    )
                    fingerprints.append(
                        {
                            name: (text_hash(row["input"]), row["generated_tokens"])
                            for name, row in io.items()
                        }
                    )
                    if profile in ("conversation", "conversation-tools"):
                        checked = 0
                        for node in dataset.nodes:
                            if node.kind != "llm" or node.predecessors[0] not in io:
                                continue
                            previous = io[node.predecessors[0]]
                            assert io[node.name]["input"].startswith(
                                previous["input"] + previous["output"]
                            ), node.name
                            checked += 1
                        assert checked == 26
                    if profile == "continuation":
                        for role, count in (("programmer", 3), ("reviser", 2)):
                            for index in range(1, count + 1):
                                previous = io[f"{role}_{index}_validate_patch"]
                                assert io[f"{role}_{index}_repair"]["input"].startswith(
                                    previous["input"] + previous["output"]
                                )
                assert all(item == fingerprints[0] for item in fingerprints)

    asyncio.run(run())


def test_join_uses_declared_order_and_keeps_shared_prefix_once():
    dataset = load_profile()
    node = next(node for node in dataset.nodes if node.name == "reviewer_1")
    outputs = {
        name: ["shared input", "shared generated text", f"branch {name}"]
        for name in reversed(node.predecessors)
    }
    chunks = compose_input(dataset, node, outputs)
    assert chunks == [
        "shared input",
        "shared generated text",
        *[f"branch {name}" for name in node.predecessors],
        dataset.templates[node.template],
    ]
    assert chunks == compose_input(dataset, node, dict(reversed(list(outputs.items()))))


@pytest.mark.parametrize(
    "defect",
    [
        "cycle",
        "missing_predecessor",
        "duplicate_node",
        "duplicate_app",
        "unknown_template",
        "zero_budget",
        "negative_tool_time",
        "unknown_tool",
        "extra_field",
    ],
)
def test_invalid_json_rejected_before_execution(defect):
    data = load_profile().model_dump()
    if defect == "cycle":
        data["nodes"][1]["predecessors"] = [data["nodes"][-1]["name"]]
    elif defect == "missing_predecessor":
        data["nodes"][1]["predecessors"] = ["missing"]
    elif defect == "duplicate_node":
        data["nodes"].append(deepcopy(data["nodes"][1]))
    elif defect == "duplicate_app":
        data["applications"].append(deepcopy(data["applications"][0]))
    elif defect == "unknown_template":
        data["nodes"][1]["template"] = "missing"
    elif defect == "zero_budget":
        data["nodes"][1]["max_tokens"] = 0
    elif defect == "negative_tool_time":
        next(node["tool"] for node in data["nodes"] if node["tool"])[
            "duration_s"
        ] = -1.0
    elif defect == "unknown_tool":
        data["applications"][0]["tool_results"]["missing"] = "result"
    else:
        data["nodes"][1]["unexpected"] = True
    with pytest.raises(ValueError):
        Dataset.model_validate(data)


def test_snapshot_copies_json_and_clients_without_modifying_source(tmp_path):
    if not SOURCE.exists():
        pytest.skip("Requires the frozen analysis checkout")
    first = materialize(
        SOURCE, tmp_path / "first", workload_profile="conversation-tools"
    )
    second = materialize(
        SOURCE, tmp_path / "second", workload_profile="conversation-tools"
    )
    assert first["materialized_helpers"] == second["materialized_helpers"]
    assert first["workload_dataset_sha256"] == second["workload_dataset_sha256"]
    assert git(SOURCE, "status", "--porcelain") == ""
    for name, expected in first["source_helpers"].items():
        assert first["materialized_helpers"][name] == expected
    graph = export_graph(tmp_path / "first")
    assert graph["source_sha256"] == first["workload_dataset_sha256"]
    assert len(graph["nodes"]) == 29
    parameters = {"num_requests": 24, "qps": [1.0, 0.5, 0.1]}
    adapter("freeze", tmp_path / "first", parameters, tmp_path / "freeze.json")
    adapter(
        "freeze",
        tmp_path / "second",
        parameters,
        tmp_path / "source-freeze.json",
        source=True,
    )
    assert json.loads((tmp_path / "source-freeze.json").read_text()) == json.loads(
        (tmp_path / "freeze.json").read_text()
    )
    assert json.loads((tmp_path / "freeze.json").read_text()) == load_profile().freeze(
        parameters["qps"]
    )
    copied = tmp_path / "first" / DATASET_NAME
    changed = json.loads(copied.read_text())
    changed["templates"]["architect_1"] += "changed"
    copied.write_text(json.dumps(changed))
    assert digest(copied) != first["materialized_helpers"][DATASET_NAME]


def test_dataset_wrapper_does_not_need_source_builder_or_tokenizer(tmp_path):
    for name in ("dataset.py", "dataset_client.py"):
        shutil.copyfile(PACKAGE / name, tmp_path / name)
    result = subprocess.run(
        [sys.executable, str(PACKAGE / "launch_client.py"), str(tmp_path), "--help"],
        cwd=tmp_path,
        env=os.environ | {"PYTHONPATH": str(tmp_path)},
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "--dataset" in result.stdout


@pytest.mark.parametrize(
    "mode",
    ["native", "agent", "offload", "offload-agent", "old-offload-agent", "mooncake"],
)
def test_each_mode_reads_the_same_snapshot_json(tmp_path, mode):
    command, _, checkout = client_command(
        Case(mode, 1.0),
        8055,
        tmp_path / "case",
        tmp_path / "launcher",
        tmp_path / "arrivals.json",
    )
    assert command[command.index("--dataset") + 1] == str(checkout / DATASET_NAME)
    assert str(PACKAGE / "launch_client.py") in command


@pytest.mark.parametrize(
    "count,rate,offsets",
    [
        (1, 1.0, None),
        (24, 0.0, None),
        (24, float("nan"), None),
        (24, 1.0, [0]),
        (24, 1.0, [0, -1] + [0] * 22),
    ],
)
def test_partial_workload_or_invalid_arrivals_do_not_send_requests(
    tmp_path, count, rate, offsets
):
    arrival = tmp_path / "arrivals.json"
    arrival.write_text(json.dumps({"offsets_s": offsets}))
    args = argparse.Namespace(
        num_requests=count,
        request_rate=rate,
        arrival_trace_file=arrival if offsets is not None else None,
    )
    with pytest.raises(ValueError):
        asyncio.run(dataset_client.benchmark(load_profile(), args))


def test_unknown_workload_profile_cannot_materialize(tmp_path):
    with pytest.raises(ValueError, match="workload profile"):
        materialize(SOURCE, tmp_path / "target", workload_profile="unknown")
