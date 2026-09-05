# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual model output after preservation, eviction, native H2D and reset."""

import concurrent.futures
import json
import time

import httpx
import pytest

from tests.tokencake.test_event_serving import post_event
from tests.tokencake.test_scheduling_serving import metric_value
from tests.tokencake.test_serving import MODEL, LocalPythonServer, annotate


@pytest.fixture(scope="module", params=[False, True])
def offload_server(request):
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
        "4",
        "--max-num-batched-tokens",
        "2048",
        "--no-scheduler-reserve-full-isl",
        "--kv-offloading-size",
        "1",
        "--kv-offloading-backend",
        "native",
        "--additional-config",
        json.dumps(
            {
                "tokencake": {
                    "scheduling": {"enabled": request.param},
                    "offload": {"min_gpu_usage": 0.0},
                }
            }
        ),
    ]
    with LocalPythonServer(
        MODEL,
        args,
        env_dict={"VLLM_USE_SIMPLE_KV_OFFLOAD": "0", "VLLM_SERVER_DEV_MODE": "1"},
    ) as server:
        yield server


def wait_metric(client, server, name, previous=0, **labels):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        metrics = client.get(server.url_for("metrics")).text
        value = metric_value(metrics, name, **labels)
        if value > previous:
            return value
        time.sleep(0.2)
    pytest.fail(
        f"Metric {name} {labels} did not increase beyond {previous}:\n{metrics}"
    )


def test_preservation_native_reload_output_and_external_reset(offload_server):
    server = offload_server
    url = server.url_for("v1/completions")
    prompt = (
        "Preserved reference.\n"
        + "This is reference context. " * 180
        + "\nWrite consecutive integers starting at 1, separated by commas.\n1,"
    )
    base = {
        "model": MODEL,
        "prompt": prompt,
        "temperature": 0,
        "seed": 42,
        "max_tokens": 128,
        "ignore_eos": True,
    }
    body = annotate(base)
    body["vllm_xargs"]["tokencake"].update(
        agent_type="preserver", offload_eligible=True, reusable_prefix=True
    )
    with httpx.Client(timeout=240) as client:
        initial = client.post(url, json=body)
        initial.raise_for_status()
        expected = initial.json()["choices"][0]["text"]
        assert "2" in expected and "3" in expected
        assert initial.json()["usage"]["completion_tokens"] == 128
        before = client.get(server.url_for("metrics")).text
        assert (
            metric_value(
                before, "vllm:kv_offload_total_bytes_total", transfer_type="GPU_to_CPU"
            )
            == 0
        )
        applied = post_event(
            client, server, body, "stall_started", estimated_duration_s=120
        )
        applied.raise_for_status()
        assert applied.json()["disposition"] == "applied"
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            work = [
                pool.submit(
                    client.post,
                    url,
                    json=annotate(base | {"prompt": f"Eviction case {i}.\n" + prompt}),
                )
                for i in range(12)
            ]
            for future in work:
                response = future.result()
                response.raise_for_status()
                result = response.json()
                assert result["usage"]["completion_tokens"] == 128
                assert result["choices"][0]["finish_reason"] == "length"
                assert "2" in result["choices"][0]["text"]
                assert "3" in result["choices"][0]["text"]
        wait_metric(client, server, "vllm:tokencake_saved_blocks_total")
        # GPU-only reset makes the origin of the successor prefix unambiguous.
        reset = client.post(
            server.url_for("reset_prefix_cache"), params={"reset_external": "false"}
        )
        reset.raise_for_status()
        returned = client.post(url, json=annotate(base))
        returned.raise_for_status()
        assert returned.json()["choices"][0]["text"] == expected
        assert returned.json()["usage"]["completion_tokens"] == 128
        wait_metric(
            client,
            server,
            "vllm:kv_offload_total_bytes_total",
            transfer_type="CPU_to_GPU",
        )
        finished = post_event(client, server, body, "stall_finished")
        finished.raise_for_status()
        assert finished.json()["disposition"] == "applied"
        reset = client.post(
            server.url_for("reset_prefix_cache"), params={"reset_external": "true"}
        )
        reset.raise_for_status()
        after_reset = client.post(url, json=annotate(base))
        after_reset.raise_for_status()
        assert after_reset.json()["choices"][0]["text"] == expected
