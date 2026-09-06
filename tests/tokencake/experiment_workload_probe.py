# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the materialized DAG in an isolated interpreter with a model test peer."""

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


def main() -> None:
    checkout = Path(sys.argv[1]).resolve()
    profile = sys.argv[2]
    os.chdir(checkout)
    sys.path.insert(0, str(checkout))
    import vllm_serving as launcher
    from agent.app.code_writer_paper_pressure import (
        CodeWriterPaperPressureApplication,
    )
    from agent.graph.meta import LLMCallMetadata, LLMTextChunkChain, TransferDataItem
    from agent.graph.node import LLMAppNode

    body_sets = []
    for mode in ("native", "agent"):
        calls: dict[str, dict] = {}

        def respond(
            request: httpx.Request, *, calls=calls, mode=mode
        ) -> httpx.Response:
            body = json.loads(request.content)
            calls[body["request_id"]] = body
            assert ("vllm_xargs" in body) == (mode == "agent")
            return httpx.Response(
                200,
                json={
                    "id": body["request_id"],
                    "choices": [
                        {
                            "text": "\ndef patch():\n    return 42\n",
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 10,
                        "total_tokens": 110,
                    },
                },
            )

        app = CodeWriterPaperPressureApplication(
            LLMCallMetadata("test", 500, 0),
            "repository context\n" * 100,
            context_token_count=4608,
            context_sources=(),
        )
        nodes = {node.name: node for node in app.graph.nodes.values()}
        assert (
            app.input_composition
            == {
                "continuation": "same-role-prefix-v1",
                "conversation": "append-only-conversation-v1",
                "conversation-tools": "append-only-conversation-v1",
            }[profile]
        )
        assert len(nodes) == 31
        assert nodes["programmer_2_validate_patch"].mcp_function.excute_time == 8.0
        assert nodes["reviser_2_validate_patch"].mcp_function.excute_time == 10.0
        assert len(app.graph.predecessors(nodes["reviewer_1"].uuid)) == 3
        if profile in ("conversation", "conversation-tools"):
            reviewer = nodes["reviewer_1"]
            branches = {
                nodes[f"code_write_{index}"].uuid: LLMTextChunkChain.from_single_text(
                    f"branch result {index}"
                )
                for index in range(1, 4)
            }
            forward = reviewer.preprocess(TransferDataItem(data=branches))
            reverse = reviewer.preprocess(
                TransferDataItem(data=dict(reversed(list(branches.items()))))
            )
            assert (
                [chunk.text for chunk in forward.chunks]
                == [chunk.text for chunk in reverse.chunks]
                == [
                    "branch result 1",
                    "branch result 2",
                    "branch result 3",
                    reviewer.system_prompt.text,
                ]
            )
        for node in nodes.values():
            if isinstance(node, launcher.McpNode):
                node.mcp_function.excute_time = 0.001
        request = launcher.ApplicationRequest("input", app)
        args = argparse.Namespace(
            port=1,
            debug=False,
            debug_sleep=False,
            record_output=True,
            disable_mcp_notifications=True,
            tokencake_mode=mode,
        )
        launcher.MODEL_PATH = "test"
        launcher.all_start_time = time.time()
        launcher.APPLICATION_INFO = {}
        launcher.IO_RECORD = {}

        async def run(request=request, args=args, respond=respond) -> None:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(respond)
            ) as client:
                launcher.HTTPX_CLIENT = client
                await launcher.send_application_request(request, 0, args)

        asyncio.run(run())
        assert launcher.APPLICATION_INFO[0]["app_finished"]
        assert len(calls) == 27
        expected_budget = 6464 if profile == "conversation-tools" else 11300
        assert sum(body["max_tokens"] for body in calls.values()) == expected_budget
        by_name = {
            name: calls[item["request_id"]]
            for name, item in launcher.IO_RECORD[0].items()
        }
        if profile == "conversation-tools":
            assert app.tool_instruction_max_tokens == 128
            compact = {
                "search_1",
                "search_2",
                "judger_1",
                "judger_2",
                "file_write_plan",
                "code_write_1",
                "code_write_2",
                "code_write_3",
                "revised_code_write_1",
                "revised_code_write_2",
                "external_eval",
                "user_confirm",
                "test_node",
            }
            for name, body in by_name.items():
                expected = (
                    128
                    if name in compact
                    else (200 if name.startswith(("architect_", "reviewer_")) else 400)
                )
                assert body["max_tokens"] == expected, name
        for role, count in (("programmer", 3), ("reviser", 2)):
            for index in range(1, count + 1):
                previous = by_name[f"{role}_{index}_validate_patch"]["prompt"]
                resumed = by_name[f"{role}_{index}_repair"]["prompt"]
                assert resumed.startswith(previous + "\ndef patch():\n    return 42\n")
        if profile in ("conversation", "conversation-tools"):
            checked = 0
            for node in nodes.values():
                if not isinstance(node, LLMAppNode):
                    continue
                predecessors = sorted(
                    app.graph.predecessors(node.uuid),
                    key=lambda key: app.graph.nodes[key].name,
                )
                previous_node = app.graph.nodes[predecessors[0]]
                if not isinstance(previous_node, LLMAppNode):
                    continue
                previous = by_name[previous_node.name]["prompt"]
                assert by_name[node.name]["prompt"].startswith(
                    previous + "\ndef patch():\n    return 42\n"
                )
                if isinstance(previous_node, launcher.McpNode):
                    assert previous_node.kvargs["reusable_prefix"]
                    assert previous_node.kvargs["offload_eligible"]
                checked += 1
            assert checked == 26
        body_sets.append(
            {
                name: (body["prompt"], body["max_tokens"])
                for name, body in by_name.items()
            }
        )
    assert body_sets[0]["architect_1"] == body_sets[1]["architect_1"]
    assert {name: item[1] for name, item in body_sets[0].items()} == {
        name: item[1] for name, item in body_sets[1].items()
    }


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        main()
    print(json.dumps({"model_calls_per_mode": 27, "identical_call_budgets": True}))
