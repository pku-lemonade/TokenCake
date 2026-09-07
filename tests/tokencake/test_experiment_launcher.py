# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protocol integration against the dataset client and a local HTTP peer."""

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TypedDict
from uuid import UUID

import aiohttp
import httpx
import pytest

from tools.tokencake_experiments.campaign import ROOT, SOURCE
from tools.tokencake_experiments.dataset import Dataset
from tools.tokencake_experiments.dataset_client import run_application as execute_app
from vllm.entrypoints.openai.completion.protocol import CompletionRequest


class Controls(TypedDict):
    fail_attempts: int
    fail_event: str


@pytest.fixture
def dataset():
    return Dataset.model_validate(
        {
            "schema_version": 1,
            "name": "test",
            "profile": "test",
            "application_class": "TestApplication",
            "application_prompt": "test",
            "max_depth": 4,
            "input_composition": "append-only-conversation-v1",
            "tool_instruction_max_tokens": None,
            "provenance": {"seed": 42},
            "templates": {"role": "Role instruction\n"},
            "nodes": [
                {
                    "name": "input",
                    "type": "input",
                    "kind": "input",
                    "predecessors": [],
                    "composition": "prepend",
                    "max_tokens": 0,
                    "metadata": {},
                },
                {
                    "name": "initial_llm_func",
                    "type": "mcp_tool_call",
                    "kind": "llm",
                    "predecessors": ["input"],
                    "composition": "append_shared",
                    "template": "role",
                    "max_tokens": 8,
                    "metadata": {"priority": 2, "depth": 1, "reusable_prefix": True},
                    "tool": {
                        "kind": "test",
                        "duration_s": 0.03,
                        "result": "fixed tool result\n",
                        "preserve_generation": True,
                    },
                },
                {
                    "name": "successor",
                    "type": "llm",
                    "kind": "llm",
                    "predecessors": ["initial_llm_func"],
                    "composition": "append_shared",
                    "template": "role",
                    "max_tokens": 8,
                    "metadata": {"depth": 2},
                },
                {
                    "name": "output",
                    "type": "output",
                    "kind": "collect",
                    "predecessors": ["successor"],
                    "composition": "prepend",
                    "max_tokens": 0,
                    "metadata": {},
                },
            ],
            "applications": [
                {
                    "id": "0",
                    "initial_input": "initial input " * 80,
                    "context_token_count": 80,
                    "context_sources": [],
                    "context_source_revision": "test",
                }
            ],
        }
    )


@pytest.fixture
def peer():
    calls = []
    controls: Controls = {"fail_attempts": 0, "fail_event": ""}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            call = {"body": body, "path": self.path, "received": time.monotonic()}
            calls.append(call)
            status = 200
            if self.path == "/v1/completions":
                completions = [entry for entry in calls if entry["path"] == self.path]
                if len(completions) <= controls["fail_attempts"]:
                    status = 503
                payload = {
                    "id": "server-prefixed-id",
                    "choices": [{"text": "answer", "finish_reason": "length"}],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 8,
                        "total_tokens": 20,
                    },
                }
            else:
                time.sleep(0.04)
                if body.get("event") == controls["fail_event"]:
                    status = 503
                payload = {"status": "applied"}
            data = json.dumps(payload).encode()
            call["returned"] = time.monotonic()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield server.server_port, calls, controls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def run_application(dataset, peer, mode, tool_duration=0.03):
    dataset.nodes[1].tool.duration_s = float(tool_duration)

    async def run():
        async with httpx.AsyncClient() as client:
            return await execute_app(
                dataset,
                dataset.applications[0],
                client=client,
                base_url=f"http://127.0.0.1:{peer[0]}/v1",
                model="test-model",
                mode=mode,
                origin=time.time(),
            )

    info, _ = asyncio.run(run())
    assert info["app_finished"]
    return info


@pytest.mark.parametrize("mode", ["native", "agent", "offload", "offload-agent"])
def test_retry_ids_bodies_full_application_and_tool_barriers(dataset, peer, mode):
    peer[2]["fail_attempts"] = 3
    info = run_application(dataset, peer, mode)
    completions = [call for call in peer[1] if call["path"] == "/v1/completions"]
    assert len(completions) == 5
    bodies = [call["body"] for call in completions]
    ids = [body["request_id"] for body in bodies]
    assert len(ids) == len(set(ids))
    assert all(
        value.startswith("tc-") and UUID(hex=value[3:]).version == 4 for value in ids
    )
    for before, after in zip(bodies[:3], bodies[1:4]):
        assert after["prompt"] == before["prompt"][: len(before["prompt"]) // 2]
    for body in bodies:
        CompletionRequest.model_validate(body)
        assert "agent_info" not in body
        if mode == "native":
            assert "vllm_xargs" not in body
        else:
            metadata = body["vllm_xargs"]["tokencake"]
            assert metadata["lifecycle_id"] == body["request_id"]
            assert "agent_name" in metadata and "application_started_at_s" in metadata
            assert (
                not {"name", "type", "app_start_time", "resume_deadline"}
                & metadata.keys()
            )
    events = [call for call in peer[1] if call["path"] != "/v1/completions"]
    tool = info["request_info"]["initial_llm_func"]
    assert tool["tool_latency"] >= 0.03
    if mode in ("offload", "offload-agent"):
        assert [call["body"]["event"] for call in events] == [
            "stall_started",
            "stall_finished",
        ]
        assert all(call["body"]["lifecycle_id"] == ids[3] for call in events)
        assert events[0]["body"]["estimated_duration_s"] == 0.03
        assert events[1]["received"] - events[0]["returned"] >= 0.03
        assert completions[-1]["received"] >= events[1]["returned"]
    else:
        assert not events


def test_four_failed_attempts_stop_application(dataset, peer):
    peer[2]["fail_attempts"] = 100
    with pytest.raises(Exception, match="after 3 retries"):
        run_application(dataset, peer, "offload-agent")
    assert len(peer[1]) == 4
    assert len({call["body"]["request_id"] for call in peer[1]}) == 4


@pytest.mark.parametrize("event", ["stall_started", "stall_finished"])
def test_event_failure_prevents_successor(dataset, peer, event):
    peer[2]["fail_event"] = event
    with pytest.raises(aiohttp.ClientResponseError, match="503"):
        run_application(dataset, peer, "offload-agent")
    assert len([call for call in peer[1] if call["path"] == "/v1/completions"]) == 1


def test_zero_tool_estimate_is_omitted(dataset, peer):
    run_application(dataset, peer, "offload-agent", tool_duration=0)
    event = next(
        call["body"] for call in peer[1] if call["body"].get("event") == "stall_started"
    )
    assert "estimated_duration_s" not in event


def test_legacy_server_uses_same_dataset_and_server_completion_id(dataset, peer):
    run_application(dataset, peer, "old-offload-agent")
    calls = peer[1]
    assert [call["path"] for call in calls] == [
        "/v1/completions",
        "/v1/mcp",
        "/v1/mcp/finished",
        "/v1/completions",
    ]
    assert calls[0]["body"]["agent_info"]["name"] == "initial_llm_func"
    assert "vllm_xargs" not in calls[0]["body"]
    assert (
        calls[1]["body"]["request_id"]
        == calls[2]["body"]["request_id"]
        == "server-prefixed-id"
    )
    assert (
        calls[3]["body"]["prompt"]
        == calls[0]["body"]["prompt"] + "answer\nfixed tool result\nRole instruction\n"
    )


def test_cli_reads_json_and_writes_complete_analyzable_results(dataset, peer, tmp_path):
    dataset.nodes[1].tool.duration_s = 0.0
    dataset.applications.append(
        dataset.applications[0].model_copy(
            update={"id": "1", "initial_input": "second fixed input"}
        )
    )
    data = tmp_path / "input.json"
    data.write_text(dataset.model_dump_json())
    output = tmp_path / "outputs.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.tokencake_experiments.dataset_client",
            "--dataset",
            str(data),
            "--port",
            str(peer[0]),
            "--model_path",
            "test",
            "--request_rate",
            "1000",
            "--task",
            "test",
            "--tokencake-mode",
            "native",
            "--output_dir",
            str(tmp_path / "apps"),
            "--output_file",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    apps_file = tmp_path / "apps/app_qps_1000.0_num_2.json"
    apps = json.loads(apps_file.read_text())
    assert len(peer[1]) == 4
    assert [row["arrival_offset_s"] for row in apps.values()] == [0.0, 0.001]
    assert all(
        row["app_finished"] and len(row["request_info"]) == 4 for row in apps.values()
    )
    records = json.loads(output.read_text())
    assert records["1"]["initial_llm_func"]["input"].startswith("second fixed input")
    assert records["1"]["successor"]["input"].startswith(
        records["1"]["initial_llm_func"]["input"] + "answer\nfixed tool result\n"
    )
    if SOURCE.exists():
        checked = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json,sys\n"
                "from tools.tokencake_experiments import analysis\n"
                "apps = json.load(open(sys.argv[1]))\n"
                "print(json.dumps(analysis.validate_dag_completion(apps, '', 2)))",
                str(apps_file),
            ],
            cwd=SOURCE,
            env=os.environ | {"PYTHONPATH": str(SOURCE)},
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert checked.returncode == 0, checked.stderr
        validation = json.loads(checked.stdout)
        assert validation["passed"]
        assert validation["terminal_status_counts"] == {
            "FINISHED_LOCAL": 4,
            "FINISHED_LENGTH_CAPPED": 4,
        }


def test_failed_branch_cancels_other_branches_before_join(dataset):
    data = dataset.model_dump()
    branch = dict(data["nodes"][1], name="other", tool=None)
    data["nodes"].append(branch)
    data["nodes"][2]["predecessors"].append("other")
    dataset = Dataset.model_validate(data)

    async def run():
        active = asyncio.Event()
        canceled = asyncio.Event()
        sent = []

        async def respond(request):
            name = json.loads(request.content)["vllm_xargs"]["tokencake"]["agent_name"]
            sent.append(name)
            if name == "other":
                active.set()
                try:
                    await asyncio.Future()
                finally:
                    canceled.set()
            await active.wait()
            return httpx.Response(503)

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(RuntimeError, match="after 3 retries"):
                await asyncio.wait_for(
                    execute_app(
                        dataset,
                        dataset.applications[0],
                        client=client,
                        base_url="http://test/v1",
                        model="test",
                        mode="agent",
                        origin=time.time(),
                    ),
                    timeout=3,
                )
        assert canceled.is_set()
        assert "successor" not in sent

    asyncio.run(run())
