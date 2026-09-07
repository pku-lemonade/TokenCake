# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-time migration of source workload data into standalone JSON datasets."""

import argparse
import contextlib
import copy
import io
import json
import os
import random
import sys
from pathlib import Path

from tools.tokencake_experiments.dataset import Dataset
from tools.tokencake_experiments.materialize import SOURCE_REVISION, git


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if git(source, "rev-parse", "HEAD") != SOURCE_REVISION or git(
        source, "status", "--porcelain"
    ):
        raise ValueError("Export requires the clean frozen source revision")
    model_path = str(args.model.resolve())
    output.mkdir(parents=True, exist_ok=False)
    import vllm.tokenizers

    sys.modules["vllm.transformers_utils.tokenizer"] = vllm.tokenizers
    os.chdir(source)
    sys.path.insert(0, str(source))
    import vllm_serving as launcher
    from agent.graph.meta import LLMCallMetadata, LLMTextChunkChain
    from agent.mcp.mcp_node import McpNode

    random.seed(42)
    launcher.MODEL_PATH = model_path
    with contextlib.redirect_stdout(io.StringIO()):
        requests = launcher.generate_requests_app(
            str(source / "dataset/agentcodeclean_new.json"),
            24,
            "code-paper-pressure",
            {"llm_metadata": LLMCallMetadata("dataset-export", 500, 0)},
            workload_source_revision="be5f23ca50818f74f485ea82f42138ec9506e1f0",
        )
    first = requests[0].application
    graph = first.graph
    nodes = {key: node for key, node in graph.nodes.items() if node.node_type != "void"}
    templates = {}
    records = []
    for key, node in nodes.items():
        model = node.type_step != 1
        predecessors = [nodes[p].name for p in graph.predecessors(key) if p in nodes]
        record = {
            "name": node.name,
            "type": node.node_type,
            "kind": "llm" if model else "collect" if predecessors else "input",
            "predecessors": predecessors,
            "template": node.name if model or predecessors else None,
            "composition": "prepend_shared"
            if model and "deduplicating" in node.input_composer.__qualname__
            else "prepend",
            "max_tokens": node.metadata.max_new_tokens if model else 0,
            "metadata": node.kvargs,
            "tool": None,
        }
        if record["template"] is not None:
            templates[node.name] = (
                node.system_prompt.text
                if model
                else "\n=== FINAL MCP TEST RESULTS ===\n"
            )
        if isinstance(node, McpNode):
            record["tool"] = {
                "kind": node.mcp_function.name,
                "duration_s": float(node.mcp_function.excute_time),
                "result": node.mcp_output(LLMTextChunkChain(chunks=[])).to_text(),
                "preserve_generation": bool(
                    node.kvargs.get("preserve_llm_output_after_tool", False)
                ),
            }
        records.append(record)
    applications = []
    for index, request in enumerate(requests):
        app = request.application
        start = next(
            node for node in app.graph.nodes.values() if node.name == "start_mcp"
        )
        applications.append(
            {
                "id": str(index),
                "initial_input": start.prompt,
                "context_token_count": app.context_token_count,
                "context_sources": list(app.context_sources),
                "context_source_revision": app.context_source_revision,
                "tool_results": {
                    node.name: node.mcp_output(LLMTextChunkChain(chunks=[])).to_text()
                    for node in app.graph.nodes.values()
                    if node.node_type == "mcp_search"
                },
            }
        )
    base = {
        "schema_version": 1,
        "name": "code-paper-pressure",
        "profile": "frozen",
        "application_class": type(first).__name__,
        "application_prompt": requests[0].prompt,
        "max_depth": max(node.kvargs.get("depth", 0) for node in graph.nodes.values()),
        "input_composition": None,
        "tool_instruction_max_tokens": None,
        "provenance": {
            "source_revision": "7a608a4e53ea990b2540c93b4d28cb795b905109",
            "seed": 42,
            "tool_results": "fixed samples; model outputs remain live",
        },
        "templates": templates,
        "nodes": records,
        "applications": applications,
    }
    for profile in ("frozen", "continuation", "conversation", "conversation-tools"):
        data = copy.deepcopy(base)
        data["profile"] = profile
        if profile == "continuation":
            data["input_composition"] = "same-role-prefix-v1"
            for node in data["nodes"]:
                if node["name"].endswith("_repair"):
                    node["composition"] = "same_role"
        elif profile in ("conversation", "conversation-tools"):
            data["input_composition"] = "append-only-conversation-v1"
            for node in data["nodes"]:
                if node["kind"] != "llm":
                    continue
                node["composition"] = "append_shared"
                node["predecessors"].sort()
                if node["tool"] and any(
                    node["name"] in successor["predecessors"]
                    and successor["kind"] == "llm"
                    for successor in data["nodes"]
                ):
                    node["tool"]["preserve_generation"] = True
                    node["metadata"].update(
                        preserve_llm_output_after_tool=True,
                        reusable_prefix=True,
                        offload_eligible=True,
                    )
                if (
                    profile == "conversation-tools"
                    and node["tool"]
                    and node["metadata"].get("stage_type")
                    not in ("branch_validation", "revision_validation")
                ):
                    node["max_tokens"] = 128
            if profile == "conversation-tools":
                data["tool_instruction_max_tokens"] = 128
        Dataset.model_validate(data)
        with (output / f"{profile}.json").open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    print(f"Exported four JSON datasets to {output}")


if __name__ == "__main__":
    main()
