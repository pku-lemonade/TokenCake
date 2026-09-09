# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audited, single-generation HTTP transport and real tool-window events."""

import hashlib
import json
import signal
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx

MODES = ("base", "agent", "offload", "agent_offload")


class ContextLimitError(RuntimeError):
    pass


class TaskDeadlineError(TimeoutError):
    pass


class RequestDeadlineError(TimeoutError):
    pass


class EventError(RuntimeError):
    pass


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


class Journal:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("x", buffering=1)

    def record(self, event: str, **fields):
        self.stream.write(
            json.dumps(
                {"event": event, "time": time.time(), "monotonic": time.monotonic()}
                | fields,
                ensure_ascii=False,
            )
            + "\n"
        )

    def close(self):
        self.stream.close()


@contextmanager
def request_alarm(seconds: float, *, error_type=RequestDeadlineError, message=None):
    """Bound the entire HTTP attempt, including connection and response reading.

    Each benchmark task runs in its own main process. There is no nested timer;
    the experiment supervisor independently enforces the application deadline.
    """
    previous = signal.getsignal(signal.SIGALRM)

    def expired(*_):
        raise error_type(message or f"HTTP attempt exceeded {seconds:.3f} seconds")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@dataclass(frozen=True)
class TaskContext:
    task_id: str
    agent_type: str
    mode: str
    arrived_at: float
    arrival_offset_s: float
    deadline_at: float

    def remaining(self) -> float:
        remaining = self.deadline_at - time.time()
        if remaining <= 0:
            raise TaskDeadlineError(f"Task {self.task_id} exceeded its time budget")
        return remaining


class Transport:
    def __init__(
        self,
        base_url: str,
        context: TaskContext,
        journal: Journal,
        *,
        max_context: int = 32768,
        client: httpx.Client | None = None,
    ):
        if context.mode not in MODES:
            raise ValueError(f"Unknown mode: {context.mode}")
        self.context, self.journal = context, journal
        self.max_context = max_context
        self.base_url = base_url.rstrip("/")
        # httpx has no implicit request retries. Do not inherit proxy variables
        # for the local service or enable any response cache.
        self.client = client or httpx.Client(trust_env=False)
        self.last_id: str | None = None
        self.previous_prompt: list[int] = []
        self.calls = 0

    @property
    def sends_events(self):
        return self.context.mode in ("offload", "agent_offload")

    def complete(
        self, endpoint: str, payload: dict, prompt_ids: list[int], *, reusable: bool
    ) -> dict:
        self.last_id = None
        self.context.remaining()
        if payload.get("n", 1) != 1 or payload.get("stream", False):
            raise ValueError("The benchmark requires one non-streaming generation")
        allowance = self.max_context - len(prompt_ids) - 2
        if allowance <= 0:
            self.journal.record("context_limit", input_tokens=len(prompt_ids))
            raise ContextLimitError(f"Input has {len(prompt_ids)} tokens")
        payload = payload | {
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "max_tokens": min(4096, allowance, payload.get("max_tokens", 4096)),
        }
        shared = 0
        for left, right in zip(self.previous_prompt, prompt_ids):
            if left != right:
                break
            shared += 1
        prompt_hash = hashlib.sha256(
            json.dumps(prompt_ids, separators=(",", ":")).encode()
        ).hexdigest()
        self.previous_prompt = prompt_ids.copy()
        self.calls += 1
        for attempt in range(3):
            identifier = "tc-" + uuid.uuid4().hex
            body = payload | {"request_id": identifier}
            if self.context.mode != "base":
                body["vllm_xargs"] = {
                    "tokencake": {
                        "lifecycle_id": identifier,
                        "agent_type": self.context.agent_type,
                        "agent_name": self.context.agent_type,
                        "application_started_at_s": self.context.arrived_at,
                        "application_start_offset_s": self.context.arrival_offset_s,
                        "application_elapsed_s": time.time() - self.context.arrived_at,
                        "offload_eligible": reusable,
                        "reusable_prefix": reusable,
                    }
                }
            timeout = min(600.0, self.context.remaining())
            self.journal.record(
                "model_attempt",
                call=self.calls,
                attempt=attempt,
                lifecycle_id=identifier,
                endpoint=endpoint,
                body=body,
                input_tokens=len(prompt_ids),
                input_sha256=prompt_hash,
                previous_input_common_tokens=shared,
                timeout_s=timeout,
            )
            started = time.monotonic()
            retryable = False
            try:
                with request_alarm(timeout):
                    response = self.client.post(
                        f"{self.base_url}/{endpoint}", json=body, timeout=timeout
                    )
                    retryable = response.status_code in (429, 503)
                    response.raise_for_status()
                    result = response.json()
                if len(result["choices"]) != 1:
                    raise ValueError("Expected exactly one response choice")
                self.last_id = identifier
                self.journal.record(
                    "model_response",
                    lifecycle_id=identifier,
                    duration_s=time.monotonic() - started,
                    response=result,
                )
                return result
            except Exception as exc:
                retryable |= isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout))
                self.journal.record(
                    "model_error",
                    lifecycle_id=identifier,
                    duration_s=time.monotonic() - started,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    retryable=retryable,
                )
                if not retryable or attempt == 2:
                    raise
                delay = float(attempt + 1)
                if self.context.remaining() <= delay:
                    raise TaskDeadlineError("Insufficient time for retry") from exc
                time.sleep(delay)
        raise AssertionError("Unreachable")

    def event(self, body: dict, *, cleanup: bool = False):
        # An event may have been applied even if its acknowledgement was lost.
        # Retrying the IDENTICAL event is supported by the server protocol.
        for attempt in range(3):
            timeout = 35.0 if cleanup else min(35.0, self.context.remaining())
            self.journal.record("event_attempt", body=body, attempt=attempt)
            try:
                with request_alarm(timeout):
                    response = self.client.post(
                        f"{self.base_url}/tokencake/events", json=body, timeout=timeout
                    )
                    response.raise_for_status()
                    result = response.json()
                if (
                    result.get("lifecycle_id") != body["lifecycle_id"]
                    or result.get("event") != body["event"]
                    or result.get("disposition")
                    not in ("applied", "duplicate", "late_finish")
                ):
                    raise EventError(f"Unexpected event acknowledgement: {result}")
                self.journal.record("event_ack", body=body, response=result)
                return
            except (httpx.HTTPError, RequestDeadlineError) as exc:
                self.journal.record(
                    "event_error", body=body, attempt=attempt, error=str(exc)
                )
                if (
                    isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code < 500
                ):
                    raise EventError(str(exc)) from exc
                if attempt == 2:
                    raise EventError(str(exc)) from exc
                time.sleep(attempt + 1)

    @contextmanager
    def tool_window(self, kind: str):
        identifier = self.last_id
        if identifier is None:
            raise EventError("Tool execution lacks a successful model request")
        self.last_id = None
        attempted_start = False
        tool_started = None
        try:
            if self.sends_events:
                attempted_start = True
                self.event(
                    {"event": "stall_started", "lifecycle_id": identifier, "kind": kind}
                )
            self.context.remaining()
            tool_started = time.monotonic()
            self.journal.record("tool_start", lifecycle_id=identifier, kind=kind)
            with request_alarm(
                self.context.remaining(),
                error_type=TaskDeadlineError,
                message=f"Tool exceeded task {self.context.task_id}'s deadline",
            ):
                yield
        finally:
            if tool_started is not None:
                self.journal.record(
                    "tool_end",
                    lifecycle_id=identifier,
                    kind=kind,
                    duration_s=time.monotonic() - tool_started,
                )
            if attempted_start:
                self.event(
                    {"event": "stall_finished", "lifecycle_id": identifier},
                    cleanup=True,
                )

    def close(self):
        self.client.close()
