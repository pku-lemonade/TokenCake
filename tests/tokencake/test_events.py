# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError

from vllm.entrypoints.openai.server_utils import (
    AuthenticationMiddleware,
    validation_exception_handler,
)
from vllm.tokencake.api_router import attach_router
from vllm.tokencake.config import TokenCakeConfig
from vllm.tokencake.events import LifecycleEventResult, StallFinished, StallStarted


def app_with_client(callback):
    app = FastAPI()
    app.state.args = SimpleNamespace(log_error_stack=False)
    app.exception_handler(RequestValidationError)(validation_exception_handler)
    attach_router(app)
    app.state.engine_client = SimpleNamespace(
        vllm_config=SimpleNamespace(_tokencake_config=TokenCakeConfig()),
        tokencake_event=callback,
    )
    return app


@pytest.mark.asyncio
async def test_http_waits_for_core_application():
    dispatched, applied = asyncio.Event(), asyncio.Event()

    async def callback(event):
        dispatched.set()
        await applied.wait()
        return LifecycleEventResult(
            event.lifecycle_id, event.event, "active", "applied"
        )

    app = app_with_client(callback)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        request = asyncio.create_task(
            client.post(
                "/v1/tokencake/events",
                json={
                    "event": "stall_started",
                    "lifecycle_id": f"tc-{uuid4().hex}",
                },
            )
        )
        await dispatched.wait()
        assert not request.done()
        applied.set()
        response = await request
    assert response.status_code == 200
    assert response.json()["disposition"] == "applied"


@pytest.mark.parametrize(
    "failure,status",
    [
        (NotImplementedError(), 503),
        (TimeoutError(), 503),
        (RuntimeError("private exception detail"), 500),
    ],
)
def test_unavailable_and_sanitized_errors(failure, status):
    async def callback(event):
        raise failure

    client = TestClient(app_with_client(callback))
    response = client.post(
        "/v1/tokencake/events",
        json={"event": "stall_finished", "lifecycle_id": f"tc-{uuid4().hex}"},
    )
    assert response.status_code == status
    assert "private exception detail" not in response.text


@pytest.mark.parametrize(
    "field,value",
    [
        ("event", "other"),
        ("lifecycle_id", "invalid"),
        ("estimated_duration_s", 0),
        ("estimated_duration_s", -1),
        ("estimated_duration_s", True),
        ("estimated_duration_s", "1"),
        ("estimated_duration_s", None),
        ("kind", ""),
        ("kind", 1),
        ("unknown", 1),
    ],
)
def test_strict_schema_http_400(field, value):
    async def callback(event):
        pytest.fail("Malformed event was dispatched")

    client = TestClient(app_with_client(callback))
    body = {"event": "stall_started", "lifecycle_id": f"tc-{uuid4().hex}"} | {
        field: value
    }
    assert client.post("/v1/tokencake/events", json=body).status_code == 400
    assert (
        client.post(
            "/v1/tokencake/events",
            content="{",
            headers={"content-type": "application/json"},
        ).status_code
        == 400
    )


def test_finish_is_strict_and_defaults_are_internal():
    identifier = f"tc-{uuid4().hex}"
    event = StallStarted(event="stall_started", lifecycle_id=identifier)
    assert event.kind == "stall" and event.estimated_duration_s is None
    with pytest.raises(ValidationError):
        TypeAdapter(StallFinished).validate_python(
            {"event": "stall_finished", "lifecycle_id": identifier, "kind": "stall"}
        )


def test_native_auth_and_disabled_path():
    async def callback(event):
        pytest.fail("Disabled event was dispatched")

    app = app_with_client(callback)
    app.add_middleware(AuthenticationMiddleware, tokens=["test-key"])
    app.state.engine_client.vllm_config._tokencake_config = None
    client = TestClient(app)
    body = {"event": "stall_started", "lifecycle_id": f"tc-{uuid4().hex}"}
    assert client.post("/v1/tokencake/events", json=body).status_code == 401
    assert (
        client.post(
            "/v1/tokencake/events",
            json=body,
            headers={"Authorization": "Bearer test-key"},
        ).status_code
        == 503
    )
