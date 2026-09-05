# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export golden decisions by executing the frozen source implementation."""

import argparse
import json
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SOURCE_REVISION = "7a608a4e53ea990b2540c93b4d28cb795b905109"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from vllm.mcp.agent_info import AgentInfo, AgentInfoManager
    from vllm.mcp.agent_scheduler import AgentScheduler

    import vllm
    from vllm.v1.core.sched.request_queue import AgentRequestQueue

    source = Path(vllm.__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == SOURCE_REVISION, (source, revision)
    rng = random.Random(42)
    now = 1000.0
    cases = []
    names = {
        "importance": "priority",
        "application_started_at_s": "app_start_time",
        "application_start_offset_s": "app_start_offset",
        "application_elapsed_s": "app_elapsed_time",
        "application_max_depth": "app_max_depth",
    }
    queue = AgentRequestQueue(AgentInfoManager())
    for i in range(32):
        metadata = {
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
        request = SimpleNamespace(
            agent_info={names.get(k, k): v for k, v in metadata.items()},
            priority=5,
            arrival_time=arrival,
        )
        with patch("time.time", return_value=now):
            score = queue.get_agent_priority(request)
        cases.append(dict(metadata=metadata, arrival=arrival, now=now, score=score))

    info = AgentInfoManager()
    manager = AgentScheduler(True, info, 1000, 16, SimpleNamespace(usage=0.8))
    histories = {}
    for i, key in enumerate(["author", "reviewer", "tester", "planner"]):
        info.agent_priority_dict[key] = float(i)
        manager._observed_priority_by_agent[key] = float(i)
        manager._waiting_count_by_agent[key] = i
        manager._waiting_avg_wait_by_agent[key] = i * 10.0
        manager.agent_deferral_count[key] = i * 2
        manager.agent_preemption_count[key] = i
        manager.agent_block_num_dict[key] = i * 10
        for j in range(2):
            info.add_agent_info(
                AgentInfo(
                    request_id=f"{key}-{j}",
                    name=key,
                    type=key,
                    input_len=100 * (j + 1),
                    output_len=20 * j,
                    time=2.0 * j,
                    depth=i,
                    out_degree=j,
                    in_degree=2,
                    app_elapsed_time=i * 5.0,
                    remaining_depth=5 - i,
                )
            )
        histories[key] = dict(
            importance=float(i),
            deferrals=i * 2,
            preemptions=i,
            samples=2,
            input_tokens=300,
            output_tokens=20,
            completed=1,
            duration=2.0,
            depth=i * 2,
            out_degree=1,
            in_degree=4,
            elapsed=i * 10.0,
            remaining=2 * (5 - i),
        )
    manager.agent_scores = {
        key: manager._compute_agent_score(key).final_score for key in histories
    }
    manager.important_agent_types = manager._select_critical_agent_types(set(histories))
    manager._update_reserve_ratio_from_gpu_usage()
    manager._partition_reserved_blocks()
    result = dict(
        source_revision=revision,
        requests=cases,
        histories=histories,
        waiting=manager._waiting_count_by_agent,
        average_wait=manager._waiting_avg_wait_by_agent,
        used=manager.agent_block_num_dict,
        scores=manager.agent_scores,
        critical=sorted(manager.important_agent_types),
        reserve_ratio=manager.reserve_ratio,
        reserved=manager.agent_reserve_num_dict,
        shared=manager.shared_gpu_blocks,
    )
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")


if __name__ == "__main__":
    main()
