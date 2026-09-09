# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual model execution and acknowledged lifecycle HTTP transitions."""

import json
import time
from uuid import uuid4

import httpx
import pytest
from prometheus_client.parser import text_string_to_metric_families

from tests.tokencake.test_serving import MODEL, LocalPythonServer, annotate, body_for

AUTH = {"Authorization": "Bearer tokencake-test-key"}


@pytest.fixture(scope="module")
def server():
    args = [
        "--enforce-eager",
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.5",
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        "4096",
        "--kv-offloading-size",
        "1",
        "--kv-offloading-backend",
        "native",
        "--additional-config",
        json.dumps({"tokencake": {}}),
        "--api-key",
        "tokencake-test-key",
    ]
    with LocalPythonServer(
        MODEL,
        args,
        env_dict={"VLLM_USE_SIMPLE_KV_OFFLOAD": "0", "VLLM_SERVER_DEV_MODE": "1"},
    ) as running:
        yield running


def event_body(body, event, **kwargs):
    return {"event": event, "lifecycle_id": body["request_id"]} | kwargs


def post_event(client, server, body, event, **kwargs):
    return client.post(
        server.url_for("v1/tokencake/events"), json=event_body(body, event, **kwargs)
    )


@pytest.mark.parametrize("route", ["completions", "chat/completions", "responses"])
def test_complete_start_finish_idempotence(server, route):
    body = annotate(body_for(route))
    with httpx.Client(headers=AUTH, timeout=120) as client:
        response = client.post(server.url_for("v1", route), json=body)
        assert response.status_code == 200, response.text
        assert post_event(client, server, body, "stall_finished").status_code == 409
        response = post_event(
            client, server, body, "stall_started", estimated_duration_s=10
        )
        assert response.status_code == 200, response.text
        assert response.json()["disposition"] == "applied"
        assert (
            post_event(
                client, server, body, "stall_started", estimated_duration_s=10
            ).json()["disposition"]
            == "duplicate"
        )
        assert (
            post_event(
                client, server, body, "stall_started", estimated_duration_s=20
            ).status_code
            == 409
        )
        response = post_event(client, server, body, "stall_finished")
        assert response.json()["state"] == "finished"
        assert response.json()["disposition"] == "applied"
        assert (
            post_event(client, server, body, "stall_finished").json()["disposition"]
            == "duplicate"
        )
        assert (
            post_event(
                client, server, body, "stall_started", estimated_duration_s=10
            ).json()["disposition"]
            == "duplicate"
        )


@pytest.mark.parametrize("early_finish", [False, True])
def test_start_before_completion_and_duplicate_generation(server, early_finish):
    body = annotate(body_for("completions")) | {"max_tokens": 96, "ignore_eos": True}
    url = server.url_for("v1/completions")
    with httpx.Client(headers=AUTH, timeout=120) as client:
        with client.stream("POST", url, json=body | {"stream": True}) as response:
            assert response.status_code == 200
            lines = response.iter_lines()
            assert any(line.startswith("data: ") for line in lines)
            applied = post_event(
                client, server, body, "stall_started", estimated_duration_s=20
            )
            assert applied.status_code == 200, applied.text
            duplicate = client.post(url, json=body)
            assert duplicate.status_code == 500, duplicate.text
            if early_finish:
                assert (
                    post_event(client, server, body, "stall_finished").json()[
                        "disposition"
                    ]
                    == "applied"
                )
            assert any(line == "data: [DONE]" for line in lines)
        result = post_event(client, server, body, "stall_finished")
        assert result.json()["disposition"] == (
            "duplicate" if early_finish else "applied"
        )
        assert result.json()["state"] == "finished"
        assert client.post(url, json=body_for("completions")).status_code == 200


@pytest.mark.parametrize("early_finish", [False, True])
def test_abort_after_start_preserves_first_terminal(server, early_finish):
    body = annotate(body_for("completions")) | {"max_tokens": 512, "ignore_eos": True}
    with httpx.Client(headers=AUTH, timeout=120) as client:
        with client.stream(
            "POST", server.url_for("v1/completions"), json=body | {"stream": True}
        ) as response:
            assert response.status_code == 200
            lines = response.iter_lines()
            assert any(line.startswith("data: ") for line in lines)
            assert (
                post_event(
                    client, server, body, "stall_started", estimated_duration_s=20
                ).status_code
                == 200
            )
            if early_finish:
                assert (
                    post_event(client, server, body, "stall_finished").status_code
                    == 200
                )
        # Closing the stream invokes the real API disconnect/EngineCore abort path.
        time.sleep(1)
        response = post_event(client, server, body, "stall_finished")
        assert response.status_code == 200, response.text
        assert response.json()["disposition"] == (
            "duplicate" if early_finish else "late_finish"
        )
        assert response.json()["state"] == ("finished" if early_finish else "aborted")


def test_auth_unknown_excluded_routes_and_native_reset(server):
    with httpx.Client(headers=AUTH, timeout=120) as client:
        url = server.url_for("v1/tokencake/events")
        body = {"event": "stall_finished", "lifecycle_id": f"tc-{uuid4().hex}"}
        assert httpx.post(url, json=body).status_code == 401
        assert client.post(url, json=body).status_code == 404
        for path in (
            "mcp",
            "mcp/finished",
            "mcp/health",
            "mcp/debug",
            "mcp/reset_prefix_cache",
        ):
            assert client.post(server.url_for("v1", path)).status_code == 404
            assert client.get(server.url_for("v1", path)).status_code == 404
        assert client.get(server.url_for("health")).status_code == 200
        request = annotate(body_for("completions"))
        assert (
            client.post(server.url_for("v1/completions"), json=request).status_code
            == 200
        )
        assert (
            post_event(
                client, server, request, "stall_started", estimated_duration_s=20
            ).status_code
            == 200
        )
        assert client.post(server.url_for("reset_prefix_cache")).status_code == 200
        assert (
            post_event(
                client, server, request, "stall_started", estimated_duration_s=20
            ).json()["disposition"]
            == "duplicate"
        )
        assert (
            client.post(
                server.url_for("reset_prefix_cache"), params={"reset_external": "true"}
            ).status_code
            == 200
        )
        assert (
            post_event(client, server, request, "stall_finished").json()["disposition"]
            == "late_finish"
        )
        metrics = client.get(server.url_for("metrics"))
        assert metrics.status_code == 200
        samples = [
            sample
            for family in text_string_to_metric_families(metrics.text)
            for sample in family.samples
            if sample.name.startswith("vllm:tokencake_")
        ]
        assert samples and any(
            s.name == "vllm:tokencake_lifecycle_total" and s.value > 0 for s in samples
        )
        assert all(set(s.labels) <= {"engine", "outcome"} for s in samples)
        assert not any(request["request_id"] in str(s.labels) for s in samples)
