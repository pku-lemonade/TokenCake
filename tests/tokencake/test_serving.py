# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-backed API correctness checks, separate from full-DAG performance gates."""

import concurrent.futures
import json
import os
import subprocess
import sys
from uuid import uuid4

import httpx
import pytest

from tests.utils import RemoteOpenAIServer

MODEL = os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


class LocalPythonServer(RemoteOpenAIServer):
    def _start_server(self, model, vllm_serve_args, env_dict):
        env = os.environ | {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"} | (env_dict or {})
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "vllm.entrypoints.cli.main",
                "serve",
                model,
                *vllm_serve_args,
            ],
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            start_new_session=True,
        )


@pytest.fixture(scope="module", params=["native", "scheduling", "offload", "both"])
def server(request):
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
    ]
    if request.param != "native":
        settings = {}
        if request.param == "scheduling":
            settings["offload"] = {"enabled": False}
        elif request.param == "offload":
            settings["scheduling"] = {"enabled": False}
        args += ["--additional-config", json.dumps({"tokencake": settings})]
    if request.param in ("offload", "both"):
        args += ["--kv-offloading-size", "1", "--kv-offloading-backend", "native"]
    with LocalPythonServer(
        MODEL, args, env_dict={"VLLM_USE_SIMPLE_KV_OFFLOAD": "0"}
    ) as running:
        yield running


def body_for(route):
    body = {"model": MODEL, "temperature": 0, "seed": 42}
    prompt = "Name the capital of France in one sentence."
    if route == "completions":
        body |= {"prompt": prompt, "max_tokens": 24}
    elif route == "chat/completions":
        body |= {"messages": [{"role": "user", "content": prompt}], "max_tokens": 24}
    else:
        body |= {"input": prompt, "max_output_tokens": 24, "store": False}
    return body


def annotate(body):
    lifecycle_id = f"tc-{uuid4().hex}"
    return body | {
        "request_id": lifecycle_id,
        "vllm_xargs": {
            "other": {"values": [1, None, False]},
            "tokencake": {
                "lifecycle_id": lifecycle_id,
                "depth": 2,
                "future": {"nested": [True, {"value": None}]},
            },
        },
    }


def generated_text(route, response):
    if route == "responses":
        return "".join(
            content["text"]
            for item in response["output"]
            if item["type"] == "message"
            for content in item["content"]
            if content["type"] == "output_text"
        )
    choice = response["choices"][0]
    return choice["text"] if route == "completions" else choice["message"]["content"]


@pytest.mark.parametrize("route", ["completions", "chat/completions", "responses"])
def test_model_output_and_concurrent_metadata(server, route):
    url = server.url_for("v1", route)
    body = body_for(route)
    with httpx.Client(timeout=120) as client:
        ordinary = client.post(url, json=body)
        ordinary.raise_for_status()
        expected = generated_text(route, ordinary.json())
        assert expected and "Paris" in expected
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            pending = [
                pool.submit(client.post, url, json=annotate(body)) for _ in range(4)
            ]
            for result in pending:
                response = result.result()
                response.raise_for_status()
                assert generated_text(route, response.json()) == expected


@pytest.mark.parametrize("route", ["completions", "chat/completions", "responses"])
def test_stream_and_request_rejection(server, route):
    url = server.url_for("v1", route)
    body = annotate(body_for(route))
    with httpx.Client(timeout=120) as client:
        with client.stream("POST", url, json=body | {"stream": True}) as response:
            response.raise_for_status()
            events = [
                json.loads(line[6:])
                for line in response.iter_lines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
        assert events
        if route == "responses":
            completed = [e for e in events if e.get("type") == "response.completed"]
            assert len(completed) == 1
            assert "Paris" in generated_text(route, completed[0]["response"])
        else:
            assert any(c.get("finish_reason") for e in events for c in e["choices"])
        invalid = body | {"request_id": f"tc-{uuid4().hex}"}
        response = client.post(url, json=invalid)
        assert response.status_code == 400, response.text
        assert "match" in response.text
        invalid = annotate(body_for(route))
        invalid["vllm_xargs"]["tokencake"]["importance"] = "invalid"
        response = client.post(url, json=invalid)
        assert response.status_code == 400, response.text


@pytest.mark.parametrize("route", ["completions", "chat/completions"])
def test_native_fanout_and_annotated_rejection(server, route):
    url = server.url_for("v1", route)
    body = body_for(route) | {"n": 2, "temperature": 0.7}
    with httpx.Client(timeout=120) as client:
        response = client.post(url, json=body)
        response.raise_for_status()
        assert len(response.json()["choices"]) == 2
        response = client.post(url, json=annotate(body))
        assert response.status_code == 400, response.text
        assert "exactly one" in response.text
        if route == "completions":
            body = body_for(route) | {"prompt": ["One plus one is", "Two plus two is"]}
            response = client.post(url, json=body)
            response.raise_for_status()
            assert len(response.json()["choices"]) == 2
            response = client.post(url, json=annotate(body))
            assert response.status_code == 400, response.text
            assert "exactly one" in response.text
