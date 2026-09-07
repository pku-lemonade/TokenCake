# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare a conversation dataset with an existing historical launcher."""

import argparse
import asyncio
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

import httpx

from tools.tokencake_experiments.dataset import load_dataset, text_hash
from tools.tokencake_experiments.dataset_client import run_application


def main() -> None:
    checkout = Path(sys.argv[1]).resolve()
    dataset = load_dataset(Path(sys.argv[2]).resolve())
    if dataset.profile not in ("conversation", "conversation-tools"):
        raise ValueError(
            "Historical frozen/continuation joins used completion order; "
            "byte equality requires a conversation profile"
        )
    import vllm.tokenizers

    sys.modules["vllm.transformers_utils.tokenizer"] = vllm.tokenizers
    os.chdir(checkout)
    sys.path.insert(0, str(checkout))
    import vllm_serving as launcher
    from agent.app.code_writer_paper_pressure import CodeWriterPaperPressureApplication
    from agent.graph.meta import LLMCallMetadata, LLMTextChunkChain

    observed = {}

    def respond(request):
        body = json.loads(request.content)
        metadata = body["vllm_xargs"]["tokencake"]
        observed[metadata["agent_name"]] = body
        return httpx.Response(
            200,
            json={
                "id": body["request_id"],
                "choices": [
                    {
                        "text": "\nLIVE_" + text_hash(body["prompt"]) + "\n",
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": body["max_tokens"],
                    "total_tokens": 100 + body["max_tokens"],
                },
            },
        )

    async def no_sleep(_):
        await asyncio.sleep(0)

    async def compare():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            for entry in dataset.applications:
                legacy_app = CodeWriterPaperPressureApplication(
                    LLMCallMetadata("test", 500, 0),
                    entry.initial_input,
                    context_token_count=entry.context_token_count,
                    context_sources=tuple(entry.context_sources),
                    context_source_revision=entry.context_source_revision,
                )
                old_nodes = {
                    node.name: node
                    for node in legacy_app.graph.nodes.values()
                    if node.node_type != "void"
                }
                assert set(old_nodes) == {node.name for node in dataset.nodes}
                for node in dataset.nodes:
                    old = old_nodes[node.name]
                    assert old.kvargs == node.metadata, node.name
                    assert {
                        legacy_app.graph.nodes[key].name
                        for key in legacy_app.graph.predecessors(old.uuid)
                        if legacy_app.graph.nodes[key].node_type != "void"
                    } == set(node.predecessors)
                    if node.tool is not None:
                        assert old.mcp_function.excute_time == node.tool.duration_s
                        result = entry.tool_results.get(node.name, node.tool.result)
                        old.mcp_output = (
                            lambda _, result=result: LLMTextChunkChain.from_single_text(
                                result
                            )
                        )
                        old.mcp_function.excute_time = 0
                request = launcher.ApplicationRequest(
                    dataset.application_prompt, legacy_app
                )
                request.frozen_workload_contract = dataset.contract(entry)
                launcher.MODEL_PATH = "test"
                launcher.all_start_time = time.time()
                launcher.APPLICATION_INFO = {}
                launcher.IO_RECORD = {}
                launcher.HTTPX_CLIENT = client
                args = argparse.Namespace(
                    port=1,
                    debug=False,
                    debug_sleep=False,
                    record_output=True,
                    disable_mcp_notifications=True,
                    tokencake_mode="agent",
                )
                observed.clear()
                await launcher.send_application_request(request, int(entry.id), args)
                assert launcher.APPLICATION_INFO[int(entry.id)]["app_finished"]
                old_io = launcher.IO_RECORD[int(entry.id)]
                old_bodies = dict(observed)
                observed.clear()
                info, new_io = await run_application(
                    dataset,
                    entry,
                    client=client,
                    base_url="http://localhost:1/v1",
                    model="test",
                    mode="agent",
                    origin=time.time(),
                    sleep=no_sleep,
                )
                assert info["app_finished"]
                assert len(new_io) == len(old_io) == 27
                for name, row in new_io.items():
                    assert row["input"] == old_io[name]["input"], (
                        f"{entry.id}/{name}: prompt differs"
                    )
                    assert row["output"] == old_io[name]["output"], (
                        f"{entry.id}/{name}: output differs"
                    )
                    assert (
                        observed[name]["max_tokens"] == old_bodies[name]["max_tokens"]
                    ), name
                    volatile = {
                        "lifecycle_id",
                        "application_started_at_s",
                        "application_start_offset_s",
                        "application_elapsed_s",
                    }
                    old_meta = old_bodies[name]["vllm_xargs"]["tokencake"]
                    new_meta = observed[name]["vllm_xargs"]["tokencake"]
                    assert {
                        key: value
                        for key, value in old_meta.items()
                        if key not in volatile
                    } == {
                        key: value
                        for key, value in new_meta.items()
                        if key not in volatile
                    }, name

    asyncio.run(compare())


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        main()
    print(
        json.dumps(
            {
                "applications": 24,
                "model_calls_per_application": 27,
                "identical_prompts_budgets_metadata": True,
            }
        )
    )
