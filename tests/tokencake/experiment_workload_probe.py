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
    os.chdir(checkout)
    sys.path.insert(0, str(checkout))
    import vllm_serving as launcher
    from agent.app.code_writer_paper_pressure import (
        CodeWriterPaperPressureApplication,
    )
    from agent.graph.meta import LLMCallMetadata

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
        assert app.input_composition == "same-role-prefix-v1"
        assert len(nodes) == 31
        assert nodes["programmer_2_validate_patch"].mcp_function.excute_time == 8.0
        assert nodes["reviser_2_validate_patch"].mcp_function.excute_time == 10.0
        assert len(app.graph.predecessors(nodes["reviewer_1"].uuid)) == 3
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
        assert sum(body["max_tokens"] for body in calls.values()) == 11300
        by_name = {
            name: calls[item["request_id"]]
            for name, item in launcher.IO_RECORD[0].items()
        }
        for role, count in (("programmer", 3), ("reviser", 2)):
            for index in range(1, count + 1):
                previous = by_name[f"{role}_{index}_validate_patch"]["prompt"]
                resumed = by_name[f"{role}_{index}_repair"]["prompt"]
                assert resumed.startswith(previous + "\ndef patch():\n    return 42\n")
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
