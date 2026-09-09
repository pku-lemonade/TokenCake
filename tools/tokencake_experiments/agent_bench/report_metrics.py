# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quality over elapsed time and explicitly scoped request diagnostics."""

from collections import Counter

QUALITY_CHECKPOINTS_S = (900, 1800, 3600)
INPUT_EDGES = (2048, 4096, 8192, 16384, 32768)
OUTPUT_EDGES = (128, 256, 512, 1024, 2048, 4096)


def completion_progress(tasks: dict, execution: dict) -> dict:
    start, wall = execution.get("started_at"), execution.get("wall_s")
    located, missing = [], []
    for task_id, row in tasks.items():
        if not row["resolved"]:
            continue
        end = row.get("ended_at")
        if start is None or end is None or end < start:
            missing.append(task_id)
        else:
            located.append({"task_id": task_id, "elapsed_s": end - start})
    located.sort(key=lambda row: (row["elapsed_s"], row["task_id"]))
    complete = (
        wall is not None
        and not execution.get("controller_error")
        and all(
            row["status"] != "terminal_missing"
            and not row["status"].startswith(("not_started", "controller_"))
            for row in tasks.values()
        )
    )
    checkpoints = []
    for seconds in QUALITY_CHECKPOINTS_S:
        covered = wall is not None and (seconds <= wall or complete)
        known = sum(row["elapsed_s"] <= seconds for row in located)
        count = known if covered and not missing else None
        checkpoints.append(
            {
                "budget_s": seconds,
                "resolved": count,
                "known_resolved_lower_bound": known,
                "resolved_per_gpu_hour": count * 3600 / seconds
                if count is not None
                else None,
                "coverage": (
                    "observed"
                    if covered and seconds <= wall
                    else "fixed_batch_finished_no_additional_tasks"
                    if covered
                    else "unobserved"
                ),
            }
        )
    return {
        "time_origin": "mode measurement start, before the first task arrival",
        "success_time": "final submitted result timestamp, graded offline",
        "planned": len(tasks),
        "measurement_wall_s": wall,
        "resolved_without_completion_time": missing,
        "checkpoints": checkpoints,
        "events": [
            row | {"cumulative_resolved": index} for index, row in enumerate(located, 1)
        ],
        "interpretation": (
            "Failures never increase the curve. All elapsed time, including failed "
            "work, stays in the fixed budget denominator. A finished finite batch "
            "stays constant; this does not predict further throughput."
        ),
    }


def length_bucket(value: int | None, edges: tuple[int, ...]) -> str:
    if value is None:
        return "unknown"
    lower = 0
    for upper in edges:
        if value < upper:
            return f"[{lower},{upper})"
        lower = upper
    return f"[{lower},inf)"


def request_records(records: list[dict]) -> list[dict]:
    endings = {
        row["lifecycle_id"]: row
        for row in records
        if row["event"] in ("model_response", "model_error")
    }
    result = []
    for row in records:
        if row["event"] != "model_attempt":
            continue
        end = endings.get(row["lifecycle_id"], {})
        usage = end.get("response", {}).get("usage") or {}
        result.append(
            {
                "lifecycle_id": row["lifecycle_id"],
                "attempt": row["attempt"],
                "input_tokens": row["input_tokens"],
                "output_tokens": usage.get("completion_tokens"),
                "status": end.get("event", "unacknowledged"),
                "duration_s": end.get("duration_s"),
            }
        )
    return result


def request_length_diagnostics(tasks: dict, wall: float | None, distribution) -> dict:
    groups = {}
    for task_id, task in tasks.items():
        for request in task["requests"]:
            key = (
                length_bucket(request["input_tokens"], INPUT_EDGES),
                length_bucket(request["output_tokens"], OUTPUT_EDGES),
            )
            groups.setdefault(key, []).append(
                request | {"task_id": task_id, "task_resolved": task["resolved"]}
            )
    rows = []
    for (input_bin, output_bin), requests in sorted(groups.items()):
        responses = [r for r in requests if r["status"] == "model_response"]
        tokens = sum(r["output_tokens"] or 0 for r in responses)
        rows.append(
            {
                "input_tokens": input_bin,
                "output_tokens": output_bin,
                "attempts": len(requests),
                "statuses": dict(Counter(r["status"] for r in requests)),
                "unique_tasks": len({r["task_id"] for r in requests}),
                "responses_from_resolved_tasks": sum(
                    r["task_resolved"] for r in responses
                ),
                "completed_request_http_e2e_s": distribution(
                    [r["duration_s"] for r in responses if r["duration_s"] is not None]
                ),
                "failed_attempt_http_time_s": sum(
                    r["duration_s"] or 0
                    for r in requests
                    if r["status"] != "model_response"
                ),
                "reported_output_tokens": tokens,
                "output_tokens_per_run_wall_s": tokens / wall if wall else None,
                "responses_per_run_wall_s": len(responses) / wall if wall else None,
            }
        )
    return {
        "input_bin_edges": list(INPUT_EDGES),
        "output_bin_edges": list(OUTPUT_EDGES),
        "bins": rows,
        "ttft_by_length": None,
        "decode_time_per_token_by_length": None,
        "missing_reason": (
            "Non-streaming journals contain HTTP response latency and token counts, "
            "but no request-level first-token or decode timestamps. Aggregate "
            "Prometheus timers cannot be assigned to length bins."
        ),
        "interpretation": (
            "Descriptive bins, not matched workloads. Successful HTTP responses "
            "can belong to failed tasks. Failed output lengths remain unknown. "
            "Bucket throughput uses the whole run wall time, including failures, "
            "and is only that bucket's contribution to total observed throughput."
        ),
    }


def server_diagnostics(observations: dict) -> dict:
    def counter(name, **labels):
        rows = [
            row
            for row in observations["counter_deltas"]
            if row["name"] == name
            and all(row["labels"].get(key) == value for key, value in labels.items())
        ]
        if not rows or any(row["delta"] is None for row in rows):
            return None
        return sum(row["delta"] for row in rows)

    def ratio(numerator, denominator):
        return (
            numerator / denominator if numerator is not None and denominator else None
        )

    timers = {}
    for label, name in {
        "ttft_s": "time_to_first_token_seconds",
        "queue_s": "request_queue_time_seconds",
        "prefill_s": "request_prefill_time_seconds",
        "decode_s_per_request": "request_decode_time_seconds",
        "decode_s_per_output_interval": "inter_token_latency_seconds",
        "tpot_s_request_weighted": "request_time_per_output_token_seconds",
    }.items():
        total = counter(f"vllm:{name}_sum")
        count = counter(f"vllm:{name}_count")
        timers[label] = {"sum": total, "count": count, "mean": ratio(total, count)}
    sources = {
        source: counter("vllm:prompt_tokens_by_source_total", source=source)
        for source in ("local_compute", "local_cache_hit", "external_kv_transfer")
    }
    total = counter("vllm:prompt_tokens_total")
    cached = counter("vllm:prompt_tokens_cached_total")
    return {
        "aggregate_timers": timers,
        "prompt_tokens": total,
        "prompt_tokens_by_source": sources,
        "actual_cached_prompt_token_fraction": ratio(cached, total),
        "cpu_reloaded_prompt_token_fraction": ratio(
            sources["external_kv_transfer"], total
        ),
        "preemptions": counter("vllm:num_preemptions_total"),
        "tokencake_recomputed_tokens": counter(
            "vllm:tokencake_scheduling_total", outcome="recomputed_tokens"
        ),
        "transfers": {
            direction: {
                "bytes": counter(
                    "vllm:kv_offload_total_bytes_total", transfer_type=direction
                ),
                "seconds": counter(
                    "vllm:kv_offload_total_time_total", transfer_type=direction
                ),
            }
            for direction in ("GPU_to_CPU", "CPU_to_GPU")
        },
        "interpretation": (
            "These server aggregates are not length controlled or task-quality "
            "controlled. Metric observation populations can differ, especially "
            "after client timeouts. Missing counters are unavailable, not zero. "
            "Prompt source accounting measures actual reuse, not lookup hits."
        ),
    }
