# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export node names and dependencies from the frozen workload implementation."""

import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path


def export_graph(checkout: Path) -> dict:
    source = checkout / "workload-dataset.json"
    if source.exists():
        from tools.tokencake_experiments.dataset import load_dataset

        dataset = load_dataset(source)
        return {
            "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "nodes": {
                node.name: {
                    "type": node.type,
                    "metadata": node.metadata,
                    "predecessors": node.predecessors,
                }
                for node in dataset.nodes
            },
        }
    sys.path.insert(0, str(checkout))
    from agent.app.code_writer_paper_pressure import CodeWriterPaperPressureApplication
    from agent.graph.meta import LLMCallMetadata

    with contextlib.redirect_stdout(io.StringIO()):
        app = CodeWriterPaperPressureApplication(
            LLMCallMetadata("graph-export", 500, 0),
            "context",
            context_token_count=4605,
            context_sources=(),
        )
    graph = app.graph
    nodes = {key: node for key, node in graph.nodes.items() if node.node_type != "void"}
    source = checkout / "agent/app/code_writer_paper_pressure.py"
    return {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "nodes": {
            node.name: {
                "type": node.node_type,
                "metadata": node.kvargs,
                "predecessors": [
                    nodes[key].name
                    for key in graph.predecessors(identifier)
                    if key in nodes
                ],
            }
            for identifier, node in nodes.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    payload = export_graph(args.checkout.resolve())
    with args.destination.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        {
            "nodes": len(payload["nodes"]),
            "edges": sum(len(n["predecessors"]) for n in payload["nodes"].values()),
        }
    )


if __name__ == "__main__":
    main()
