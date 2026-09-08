# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import time
from uuid import UUID

import httpx
import pytest

from tools.tokencake_experiments.agent_bench.transport import (
    ContextLimitError,
    EventError,
    Journal,
    TaskContext,
    Transport,
)


@pytest.fixture
def build(tmp_path):
    resources: list[tuple[Transport, Journal]] = []

    def factory(handler, mode="agent_offload"):
        journal = Journal(tmp_path / f"requests-{len(resources)}.jsonl")
        context = TaskContext("task", "bfcl", mode, time.time(), 0.0, time.time() + 600)
        client = httpx.Client(transport=httpx.MockTransport(handler))
        transport = Transport("http://localhost/v1", context, journal, client=client)
        resources.append((transport, journal))
        return transport

    yield factory
    for transport, journal in resources:
        transport.close()
        journal.close()


def completion():
    return httpx.Response(200, json={"choices": [{"text": "[tool()]"}]})


def ack(body):
    return httpx.Response(200, json=body | {"disposition": "applied", "state": "ok"})


@pytest.mark.parametrize("mode", ["base", "agent", "offload", "agent_offload"])
def test_metadata_and_acknowledged_tool_order(build, mode):
    observed = []
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        observed.append(body.get("event", "model"))
        return ack(body) if "event" in body else completion()

    transport = build(handler, mode)
    transport.complete("completions", {"prompt": "preserve me"}, [1, 2], reusable=True)
    with transport.tool_window("shell"):
        observed.append("actual_tool")
    if mode in ("offload", "agent_offload"):
        assert observed == ["model", "stall_started", "actual_tool", "stall_finished"]
    else:
        assert observed == ["model", "actual_tool"]
    first = bodies[0]
    assert first["prompt"] == "preserve me"
    assert first["max_tokens"] == 4096
    assert UUID(first["request_id"][3:]).version == 4
    if mode == "base":
        assert "vllm_xargs" not in first
    else:
        metadata = first["vllm_xargs"]["tokencake"]
        assert metadata["lifecycle_id"] == first["request_id"]
        assert "depth" not in metadata
        assert "estimated_duration_s" not in metadata


def test_rejected_request_gets_new_identity_without_shortening(build, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(503) if len(seen) < 3 else completion()

    transport = build(handler)
    transport.complete("completions", {"prompt": "full history"}, [1], reusable=True)
    assert len({body["request_id"] for body in seen}) == 3
    assert all(body["prompt"] == "full history" for body in seen)


def test_submitted_read_timeout_is_not_retried(build):
    requests = []

    def handler(request):
        requests.append(request)
        raise httpx.ReadTimeout("submitted request lost")

    transport = build(handler)
    with pytest.raises(httpx.ReadTimeout):
        transport.complete("completions", {}, [1], reusable=True)
    assert len(requests) == 1
    with pytest.raises(EventError), transport.tool_window("shell"):
        pytest.fail("A failed model request must never execute a tool")


def test_tool_exception_finishes_window_and_cannot_replay(build):
    events = []

    def handler(request):
        body = json.loads(request.content)
        if "event" in body:
            events.append(body["event"])
            return ack(body)
        return completion()

    transport = build(handler)
    transport.complete("completions", {}, [1], reusable=True)
    with (
        pytest.raises(ValueError, match="tool failed"),
        transport.tool_window("shell"),
    ):
        raise ValueError("tool failed")
    assert events == ["stall_started", "stall_finished"]
    with pytest.raises(EventError), transport.tool_window("shell"):
        pytest.fail("An attempt may own only one tool window")


def test_lost_event_ack_retries_identical_body(build, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    starts = []

    def handler(request):
        body = json.loads(request.content)
        if body.get("event") == "stall_started":
            starts.append(request.content)
            if len(starts) == 1:
                raise httpx.ReadTimeout("ack lost")
        return ack(body) if "event" in body else completion()

    transport = build(handler)
    transport.complete("completions", {}, [1], reusable=True)
    with transport.tool_window("shell"):
        pass
    assert len(starts) == 2 and starts[0] == starts[1]


def test_context_budget_tightens_output_and_overflow_sends_nothing(build):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return completion()

    transport = build(handler)
    transport.max_context = 10
    transport.complete("completions", {}, list(range(7)), reusable=True)
    assert bodies[0]["max_tokens"] == 1
    with pytest.raises(ContextLimitError):
        transport.complete("completions", {}, list(range(8)), reusable=True)
    assert len(bodies) == 1
