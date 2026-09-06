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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    checkout, destination = args.checkout.resolve(), args.destination.resolve()
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
    payload = {
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
    with destination.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        {
            "nodes": len(nodes),
            "edges": sum(len(n["predecessors"]) for n in payload["nodes"].values()),
        }
    )


if __name__ == "__main__":
    main()
