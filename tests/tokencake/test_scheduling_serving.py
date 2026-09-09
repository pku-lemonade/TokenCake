# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real generation under KV contention, including native preemption recovery."""

import concurrent.futures
import json

import httpx
import pytest
from prometheus_client.parser import text_string_to_metric_families

from tests.tokencake.test_serving import MODEL, LocalPythonServer, annotate


@pytest.fixture(scope="module", params=["fcfs", "priority"])
def pressure_server(request):
    args = [
        "--enforce-eager",
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.5",
        "--num-gpu-blocks-override",
        "320",
        "--block-size",
        "16",
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        "2048",
        "--scheduling-policy",
        request.param,
        "--no-scheduler-reserve-full-isl",
        "--additional-config",
        json.dumps(
            {
                "tokencake": {
                    "scheduling": {"decode_prefill_token_budget": 512},
                    "offload": {"enabled": False},
                }
            }
        ),
    ]
    with LocalPythonServer(MODEL, args) as server:
        yield server


def metric_value(text, name, **labels):
    return sum(
        sample.value
        for family in text_string_to_metric_families(text)
        for sample in family.samples
        if sample.name == name
        and all(sample.labels.get(k) == v for k, v in labels.items())
    )


def test_mixed_contended_generation_and_recovery(pressure_server):
    url = pressure_server.url_for("v1", "completions")
    bodies = [
        {
            "model": MODEL,
            "prompt": (
                f"Case {i}.\n"
                + "This is reference context. " * 180
                + "\nWrite consecutive integers starting at 1, separated by commas.\n1,"
            ),
            "temperature": 0,
            "seed": 42,
            "max_tokens": 128,
            "ignore_eos": True,
        }
        for i in range(12)
    ]
    with httpx.Client(timeout=240) as client:
        reference = client.post(url, json=bodies[0])
        reference.raise_for_status()
        expected = reference.json()["choices"][0]["text"]
        assert "2" in expected and "3" in expected
        before = client.get(pressure_server.url_for("metrics")).text
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(bodies)) as pool:
            futures = []
            for i, body in enumerate(bodies):
                if i % 3:
                    body = annotate(body)
                    body["vllm_xargs"]["tokencake"].update(
                        agent_type=f"type-{i % 2}", importance=float(i * 100)
                    )
                futures.append(pool.submit(client.post, url, json=body))
            for future in futures:
                response = future.result()
                response.raise_for_status()
                result = response.json()
                assert result["choices"][0]["finish_reason"] == "length"
                assert result["usage"]["completion_tokens"] == 128
                assert result["usage"]["prompt_tokens"] > 768
                assert "2" in result["choices"][0]["text"]
                assert "3" in result["choices"][0]["text"]
        after = client.get(pressure_server.url_for("metrics")).text
        assert metric_value(
            after, "vllm:tokencake_scheduling_total", outcome="prefill_capped"
        ) > metric_value(
            before, "vllm:tokencake_scheduling_total", outcome="prefill_capped"
        )
        assert (
            metric_value(
                after,
                "vllm:tokencake_scheduling_total",
                outcome="reservation_preempted",
            )
            == 0
        )
        # After the pressure wave, ordinary work must reproduce the serial result.
        recovered = client.post(url, json=bodies[0])
        recovered.raise_for_status()
        assert recovered.json()["choices"][0]["text"] == expected
