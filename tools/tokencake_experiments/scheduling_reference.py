# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export diagnostic decisions from the current repository scheduler."""

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from tools.tokencake_experiments.provenance import repository
    from vllm.tokencake.protocol import TokenCakeMetadata
    from vllm.tokencake.scheduling import (
        AgentHistory,
        partition_capacity,
        request_score,
    )

    revision = repository(ROOT)["commit"]
    rng = random.Random(42)
    now = 1000.0
    cases = []
    for i in range(32):
        metadata = {
            "lifecycle_id": "tc-" + UUID(int=i, version=4).hex,
            "importance": rng.choice([0.0, 1.0, 20.0, 100.0]),
            "depth": rng.randrange(9),
            "out_degree": rng.randrange(5),
            "in_degree": rng.randrange(5),
            "similarity": rng.choice([0.0, 0.125, 0.5, 1.0]),
            "application_started_at_s": rng.choice([0.0, 700.0, 950.0]),
            "application_elapsed_s": rng.choice([0.0, 45.0, 400.0]),
            "application_start_offset_s": rng.choice([0.0, 25.0, 100.0]),
            "application_max_depth": 10,
            "remaining_depth": rng.randrange(10),
            "critical_path": bool(i % 2),
            "near_completion": bool(i % 3),
            "join_group": "join" if i % 2 else "",
            "dependency_depth": rng.randrange(8),
            "fanout_width": rng.randrange(1, 9),
            "memory_weight": rng.choice([0.0, 1.0, 2.5]),
        }
        arrival = rng.choice([500.0, 990.0, 1005.0])
        score = request_score(TokenCakeMetadata(**metadata), arrival, now)
        cases.append(dict(metadata=metadata, arrival=arrival, now=now, score=score))

    histories = {
        name: AgentHistory(
            importance=float(i),
            samples=2,
            input_tokens=300,
            output_tokens=20,
            completed=1,
            duration=2.0,
            deferrals=i * 2,
            preemptions=i,
            depth=i * 2,
        )
        for i, name in enumerate(("author", "reviewer", "tester", "planner"))
    }
    scores = {
        name: history.score(history.importance, i, i * 10.0)
        for i, (name, history) in enumerate(histories.items())
    }
    plan = partition_capacity(
        scores,
        {name: history.importance for name, history in histories.items()},
        {name: i * 10 for i, name in enumerate(histories)},
        1000,
        0.2,
        0.5,
    )
    result = {
        "source_revision": revision,
        "requests": cases,
        "histories": {name: asdict(history) for name, history in histories.items()},
        "scores": scores,
        "critical": sorted(plan.critical),
        "reserved": plan.reserved,
        "shared": plan.shared,
    }
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")


if __name__ == "__main__":
    main()
