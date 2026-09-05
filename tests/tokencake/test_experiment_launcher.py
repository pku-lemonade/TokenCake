# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protocol integration against the actual frozen launcher and a local HTTP peer."""

import argparse
import asyncio
import importlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TypedDict
from uuid import UUID

import aiohttp
import httpx
import pytest

from tools.tokencake_experiments.campaign import SOURCE
from tools.tokencake_experiments.materialize import git, materialize
from vllm.entrypoints.openai.completion.protocol import CompletionRequest


class Controls(TypedDict):
    fail_attempts: int
    fail_event: str


@pytest.fixture(scope="module")
def checkout(tmp_path_factory):
    if not SOURCE.exists():
        pytest.skip("Requires the frozen source checkout")
    root = tmp_path_factory.mktemp("source-launcher")
    first = materialize(SOURCE, root / "first")
    second = materialize(SOURCE, root / "second")
    assert first["materialized_helpers"] == second["materialized_helpers"]
    assert first["patch_sha256"] == second["patch_sha256"]
    assert first["source_helpers"] != first["materialized_helpers"]
    assert git(SOURCE, "status", "--porcelain") == ""
    return root / "first"


@pytest.fixture
def launcher(checkout, monkeypatch):
    monkeypatch.chdir(checkout)
    monkeypatch.syspath_prepend(str(checkout))
    module = importlib.import_module("vllm_serving")
    assert module.__file__ is not None and str(checkout) in module.__file__
    monkeypatch.setattr(module, "APPLICATION_INFO", {})
    monkeypatch.setattr(module, "IO_RECORD", {})
    monkeypatch.setattr(module, "all_start_time", time.time(), raising=False)
    monkeypatch.setattr(module, "MODEL_PATH", "test-model", raising=False)
    return module


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
                if body["event"] == controls["fail_event"]:
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


def run_application(launcher, peer, monkeypatch, mode, tool_duration=0.03):
    from agent.app.single_agent_mcp import SingleAgentMcpApplication

    application = SingleAgentMcpApplication(
        launcher.LLMCallMetadata(model="test-model", max_new_tokens=8, temperature=0)
    )
    for node in application.graph.nodes.values():
        if isinstance(node, launcher.McpNode):
            node.mcp_function.excute_time = tool_duration
    request = launcher.ApplicationRequest("initial input " * 80, application)
    args = argparse.Namespace(
        port=peer[0],
        debug=False,
        debug_sleep=False,
        record_output=True,
        disable_mcp_notifications=mode != "offload-agent",
        tokencake_mode=mode,
    )
    for variable in ("MCP_URL", "MCP_FINISHED_URL"):
        monkeypatch.setattr(
            launcher,
            variable,
            f"http://127.0.0.1:{peer[0]}/v1/tokencake/events",
            raising=False,
        )

    async def run():
        async with httpx.AsyncClient() as client:
            monkeypatch.setattr(launcher, "HTTPX_CLIENT", client, raising=False)
            await launcher.send_application_request(request, 0, args)

    asyncio.run(run())
    assert launcher.APPLICATION_INFO[0]["app_finished"]
    return launcher.APPLICATION_INFO[0]


@pytest.mark.parametrize("mode", ["native", "agent", "offload-agent"])
def test_retry_ids_bodies_full_application_and_tool_barriers(
    launcher, peer, monkeypatch, mode
):
    peer[2]["fail_attempts"] = 3
    info = run_application(launcher, peer, monkeypatch, mode)
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
    if mode == "offload-agent":
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


def test_four_failed_attempts_stop_application(launcher, peer, monkeypatch):
    peer[2]["fail_attempts"] = 100
    with pytest.raises(Exception, match="after 3 retries"):
        run_application(launcher, peer, monkeypatch, "offload-agent")
    assert len(peer[1]) == 4
    assert len({call["body"]["request_id"] for call in peer[1]}) == 4


@pytest.mark.parametrize("event", ["stall_started", "stall_finished"])
def test_event_failure_prevents_successor(launcher, peer, monkeypatch, event):
    peer[2]["fail_event"] = event
    with pytest.raises(aiohttp.ClientResponseError, match="503"):
        run_application(launcher, peer, monkeypatch, "offload-agent")
    assert len([call for call in peer[1] if call["path"] == "/v1/completions"]) == 1


def test_zero_tool_estimate_is_omitted(launcher, peer, monkeypatch):
    run_application(launcher, peer, monkeypatch, "offload-agent", tool_duration=0)
    event = next(
        call["body"] for call in peer[1] if call["body"].get("event") == "stall_started"
    )
    assert "estimated_duration_s" not in event
