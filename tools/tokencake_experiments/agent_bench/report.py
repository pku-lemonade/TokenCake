# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Join quality, failed tasks, actual model work, and pressure observations."""

import argparse
import json
from collections import Counter
from pathlib import Path

from tools.tokencake_experiments.runtime import metric_values

from .experiment import tasks_for
from .inputs import digest, read_jsonl
from .report_metrics import (
    completion_progress,
    request_length_diagnostics,
    request_records,
    server_diagnostics,
)
from .throughput import (
    output_throughput,
    read_metric_samples,
    shared_measurement_start,
    throughput_comparisons,
)
from .transport import write_json


def percentile(values: list[float], fraction: float):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: list[float]) -> dict:
    return {
        "count": len(values),
        "sum": sum(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
    }


def optional_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def observations(service: Path, execution: dict, metric_samples=None) -> dict:
    start, end = execution.get("started_at", 0), execution.get("ended_at", 0)
    if metric_samples is None:
        metric_samples, _ = read_metric_samples(service)
    candidates = [
        sample
        for sample in metric_samples
        if start <= sample["timestamp"] <= end and sample.get("metrics")
    ]
    last = (
        max(candidates, key=lambda sample: sample["timestamp"]) if candidates else None
    )
    baseline = service / "metrics-start.prom"
    counters = []
    if baseline.exists() and last is not None:

        def samples(rows):
            return {
                (row["name"], json.dumps(row["labels"], sort_keys=True)): row["value"]
                for row in rows
                if row["name"].endswith(("_total", "_sum", "_count"))
            }

        before = samples(metric_values(baseline.read_text()))
        after = samples(last["metrics"])
        for (name, labels), value in after.items():
            difference = value - before.get((name, labels), 0.0)
            counters.append(
                {
                    "name": name,
                    "labels": json.loads(labels),
                    "delta": difference if difference >= 0 else None,
                    "counter_reset": difference < 0,
                }
            )
    peaks: dict[str, float] = {}
    if metric_samples:
        for sample in metric_samples:
            if not start <= sample["timestamp"] <= end:
                continue
            totals = Counter()
            for row in sample.get("metrics", []):
                if row["name"] in (
                    "vllm:num_requests_waiting",
                    "vllm:num_requests_running",
                    "vllm:kv_cache_usage_perc",
                    "vllm:tokencake_active_lifecycles",
                ):
                    totals[row["name"]] += row["value"]
            for name, value in totals.items():
                peaks[name] = max(peaks.get(name, value), value)
    resources = []
    path = service / "resources.jsonl"
    if path.exists():
        resources = [
            row
            for row in read_jsonl(path)
            if start <= row["timestamp"] <= end and "cgroup" in row
        ]
    memory = [
        int(row["cgroup"]["memory.current"])
        for row in resources
        if row["cgroup"].get("memory.current") is not None
    ]
    return {
        "counter_deltas": counters,
        "counter_window": {
            "source": str(service / "metrics.jsonl"),
            "cutoff": end,
            "last_sample_timestamp": last["timestamp"] if last else None,
            "sample_lag_s": end - last["timestamp"] if last else None,
        },
        "sampled_gauge_peaks": peaks,
        "sampled_cgroup_memory_peak_bytes": max(memory) if memory else None,
        "resource_samples": len(resources),
        "resource_monitor": optional_json(service / "resources-monitor.json"),
        "metrics_monitor": optional_json(service / "metrics-monitor.json"),
        "missing_metrics_are": "unavailable, not zero",
    }


def task_observations(path: Path, terminal: dict) -> dict:
    journal = path / "journal.jsonl"
    records, corrupt_lines = [], []
    if journal.exists():
        for number, line in enumerate(journal.read_text().splitlines(), 1):
            try:
                records.append(json.loads(line))
            except ValueError:
                # A forced process stop can interrupt a large JSONL write.
                # Preserve valid observations and explicitly flag lost data.
                corrupt_lines.append(number)
    attempts = [row for row in records if row["event"] == "model_attempt"]
    responses = [row for row in records if row["event"] == "model_response"]
    errors = [row for row in records if row["event"] == "model_error"]
    tools = [row["duration_s"] for row in records if row["event"] == "tool_end"]
    usage = [row["response"].get("usage") for row in responses]
    return {
        "journal_present": journal.exists(),
        "journal_corrupt_lines": corrupt_lines,
        "status": terminal["status"],
        "e2e_s": terminal.get("e2e_s"),
        "arrived_at": terminal.get("arrived_at"),
        "ended_at": terminal.get("ended_at"),
        "requests": request_records(records),
        "model_attempts": len(attempts),
        "model_responses": len(responses),
        "model_errors": len(errors),
        "retries": sum(row["attempt"] > 0 for row in attempts),
        "response_usage_missing": sum(item is None for item in usage),
        "reported_input_tokens": sum(
            item.get("prompt_tokens", 0) for item in usage if item
        ),
        "reported_output_tokens": sum(
            item.get("completion_tokens", 0) for item in usage if item
        ),
        "attempt_declared_output_tokens": sum(
            row["body"]["max_tokens"] for row in attempts
        ),
        "unacknowledged_attempts": len(attempts) - len(responses),
        "model_time_s": sum(row["duration_s"] for row in responses + errors),
        "tool_times_s": tools,
        "event_attempts": sum(row["event"] == "event_attempt" for row in records),
        "event_acks": sum(row["event"] == "event_ack" for row in records),
        "previous_input_common_tokens": [
            row["previous_input_common_tokens"]
            for row in attempts
            if row["attempt"] == 0
        ],
    }


def build_report(experiment: Path, scores_path: Path) -> dict:
    config = json.loads((experiment / "experiment.json").read_text())
    scores = json.loads(scores_path.read_text())
    manifest_path = Path(config["manifest"])
    if (
        digest(manifest_path) != config["manifest_sha256"]
        or scores["manifest_sha256"] != config["manifest_sha256"]
        or Path(scores["experiment"]).resolve() != experiment.resolve()
    ):
        raise ValueError("Predictions, scores and frozen inputs do not match")
    manifest = json.loads(manifest_path.read_text())
    common_start = shared_measurement_start(config)
    tasks = tasks_for(manifest, config["benchmark"], config["subset"])
    ids = [task.get("instance_id", task.get("id")) for task in tasks]
    modes = {}
    for mode in config["modes"]:
        score_rows = scores["modes"][mode]["results"]
        task_scores = {row["task_id"]: row for row in score_rows}
        if len(score_rows) != len(ids) or set(task_scores) != set(ids):
            raise ValueError("Every planned task needs exactly one scoring record")
        rows = {}
        for task_id in ids:
            path = experiment / mode / "tasks" / task_id
            terminal = optional_json(path / "terminal.json") or {
                "status": "terminal_missing"
            }
            rows[task_id] = task_observations(path, terminal) | {
                "resolved": bool(task_scores[task_id]["resolved"]),
                "score_status": task_scores[task_id]["status"],
            }
        execution = optional_json(experiment / mode / "tasks/execution.json")
        service = experiment / mode / "service"
        metric_samples, metric_issues = read_metric_samples(service)
        resolved = sum(row["resolved"] for row in rows.values())
        e2e = [row["e2e_s"] for row in rows.values() if row["e2e_s"] is not None]
        modes[mode] = {
            "planned": len(ids),
            "resolved": resolved,
            "quality_percent": 100 * resolved / len(ids),
            "statuses": dict(Counter(row["status"] for row in rows.values())),
            "score_statuses": dict(
                Counter(row["score_status"] for row in rows.values())
            ),
            "all_terminal_e2e_s": distribution(e2e),
            "without_e2e": len(ids) - len(e2e),
            "without_journal": sum(not row["journal_present"] for row in rows.values()),
            "with_corrupt_journal": sum(
                bool(row["journal_corrupt_lines"]) for row in rows.values()
            ),
            "wall_s": execution.get("wall_s"),
            "resolved_per_s": resolved / execution["wall_s"]
            if execution.get("wall_s")
            else None,
            "model_work": {
                key: sum(row[key] for row in rows.values())
                for key in (
                    "model_attempts",
                    "model_responses",
                    "model_errors",
                    "retries",
                    "reported_input_tokens",
                    "reported_output_tokens",
                    "response_usage_missing",
                    "unacknowledged_attempts",
                    "attempt_declared_output_tokens",
                    "model_time_s",
                    "event_attempts",
                    "event_acks",
                )
            },
            "tool_time_s": distribution(
                [value for row in rows.values() for value in row["tool_times_s"]]
            ),
            "observations": observations(service, execution, metric_samples),
            "tasks": rows,
        }
        groups = {}
        for task in tasks:
            key = task.get("instance_id", task.get("id"))
            group = (
                key.rsplit("_", 1)[0] if config["benchmark"] == "bfcl" else task["repo"]
            )
            groups.setdefault(group, []).append(rows[key])
        modes[mode]["quality_by_category_or_repo"] = {
            group: {
                "planned": len(items),
                "resolved": sum(item["resolved"] for item in items),
                "quality_percent": 100
                * sum(item["resolved"] for item in items)
                / len(items),
                "statuses": dict(Counter(item["status"] for item in items)),
                "score_statuses": dict(Counter(item["score_status"] for item in items)),
            }
            for group, items in groups.items()
        }
        modes[mode]["correct_completion_progress"] = completion_progress(
            rows, execution
        )
        modes[mode]["request_length_diagnostics"] = request_length_diagnostics(
            rows, execution.get("wall_s"), distribution
        )
        modes[mode]["server_diagnostics"] = server_diagnostics(
            modes[mode]["observations"]
        )
        modes[mode]["output_throughput"] = output_throughput(
            service,
            execution,
            modes[mode]["model_work"]["reported_output_tokens"],
            manifest.get("model_path"),
            common_start["timestamp"],
            metric_samples,
            metric_issues,
        )
    pairs = {}
    if "base" in modes:
        for mode, data in modes.items():
            if mode == "base":
                continue
            counts = Counter(
                (
                    modes["base"]["tasks"][key]["resolved"],
                    data["tasks"][key]["resolved"],
                )
                for key in ids
            )
            pairs[mode] = {
                "both_resolved": counts[(True, True)],
                "base_only": counts[(True, False)],
                "variant_only": counts[(False, True)],
                "neither": counts[(False, False)],
                "quality_difference_pp": data["quality_percent"]
                - modes["base"]["quality_percent"],
            }
            paired = []
            for key in ids:
                left, right = modes["base"]["tasks"][key], data["tasks"][key]
                if left["resolved"] and right["resolved"]:
                    a, b = left["e2e_s"], right["e2e_s"]
                    paired.append(
                        {
                            "task_id": key,
                            "base_e2e_s": a,
                            "variant_e2e_s": b,
                            "base_over_variant": a / b if a is not None and b else None,
                        }
                    )
            pairs[mode]["both_resolved_e2e"] = {
                "tasks": paired,
                "base_over_variant": distribution(
                    [
                        row["base_over_variant"]
                        for row in paired
                        if row["base_over_variant"] is not None
                    ]
                ),
                "interpretation": (
                    "Conditional on both runs resolving the task; report exclusive "
                    "successes and all failures alongside this subset. "
                    "Empty is unavailable."
                ),
            }
            pairs[mode]["exclusive_success_ids"] = {
                "base_only": [
                    key
                    for key in ids
                    if modes["base"]["tasks"][key]["resolved"]
                    and not data["tasks"][key]["resolved"]
                ],
                "variant_only": [
                    key
                    for key in ids
                    if data["tasks"][key]["resolved"]
                    and not modes["base"]["tasks"][key]["resolved"]
                ],
            }
    return {
        "experiment": str(experiment),
        "scores": str(scores_path),
        "benchmark": config["benchmark"],
        "subset": config["subset"],
        "workers": manifest["pilot"]["workers"],
        "arrival": manifest["pilot"]["arrival"],
        "service_execution": manifest["pilot"].get("service_execution", "sequential"),
        "gpu_by_mode": manifest["pilot"].get("gpu_by_mode"),
        "host_resource_scope": (
            "CPU quota and cgroup memory are shared by both modes; "
            "per-service cgroup samples must not be added or attributed to one mode"
            if manifest["pilot"].get("service_execution") == "parallel_pair"
            else "one active model service"
        ),
        "modes": modes,
        "shared_measurement_start": common_start,
        "output_throughput_comparisons_against_base": throughput_comparisons(modes),
        "paired_quality_against_base": pairs,
        "interpretation": [
            "All planned tasks remain in quality denominators, including failures.",
            "E2E distributions include failed arrivals; "
            "unstarted tasks are counted separately.",
            "Response token counts cannot recover output "
            "from failed/unacknowledged requests.",
            "Pilot results do not establish quality equivalence; "
            "formal statistics need freezing.",
            "Primary comparisons use correct completions at a common time budget. "
            "All-terminal latency is diagnostic and can improve when tasks fail early.",
            "Closed-loop runs share order and concurrency; "
            "actual arrival times and generated work can differ.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.experiment, args.scores)
    write_json(args.output, report)
    print(
        json.dumps(
            {
                mode: {
                    key: value
                    for key, value in data.items()
                    if key not in ("tasks", "observations")
                }
                for mode, data in report["modes"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
