# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extract comparable full-DAG measurements, preserving unavailable metrics."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from prometheus_client.parser import text_string_to_metric_families


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metric(samples, name, **labels):
    values = [
        sample["value"]
        for sample in samples
        if sample["name"] == name
        and all(sample["labels"].get(key) == value for key, value in labels.items())
    ]
    return sum(values) if values else None


def percentile(values, quantile):
    if not values or not 0 <= quantile <= 1:
        raise ValueError("A percentile requires observations and a quantile in [0, 1]")
    values = sorted(values)
    position = (len(values) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (position - low) * (values[high] - values[low])


def weighted_samples(points, start, end, accessor):
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("Invalid measurement interval")
    points = [point for point in points if "monotonic" in point]
    if any(a["monotonic"] >= b["monotonic"] for a, b in zip(points, points[1:])):
        raise ValueError("Samples must have strictly increasing monotonic timestamps")
    area = duration = 0.0
    observed = []
    for index, point in enumerate(points):
        value = accessor(point)
        if value is None or not math.isfinite(value):
            continue
        next_time = points[index + 1]["monotonic"] if index + 1 < len(points) else end
        left, right = max(start, point["monotonic"]), min(end, next_time)
        if right <= left or right - left > 10:
            continue
        area += value * (right - left)
        duration += right - left
        observed.append(value)
    return {
        "mean": area / duration if duration else None,
        "maximum": max(observed) if observed else None,
        "coverage_fraction": duration / (end - start),
    }


def critical_path(app, graph):
    nodes = app["request_info"]
    predecessors = {
        name for node in graph["nodes"].values() for name in node["predecessors"]
    }
    sinks = set(graph["nodes"]) - predecessors
    name = max(sinks, key=lambda key: nodes[key]["end_time"])
    chain = []
    gaps = app["app_end_time"] - nodes[name]["end_time"]
    while True:
        if name in chain:
            raise ValueError("Cycle in workload graph")
        chain.append(name)
        predecessors = graph["nodes"][name]["predecessors"]
        if not predecessors:
            gaps += nodes[name]["start_time"] - app["app_start_time"]
            break
        previous = max(predecessors, key=lambda key: nodes[key]["end_time"])
        gap = nodes[name]["start_time"] - nodes[previous]["end_time"]
        if gap < -1e-5:
            raise ValueError(f"Node started before dependency completed: {name}")
        gaps += gap
        name = previous
    totals = {
        key: sum(nodes[name][key] for name in chain)
        for key in ("llm_latency", "tool_latency", "residual_latency")
    }
    totals["orchestration_gap_s"] = gaps
    total = sum(totals.values())
    if abs(total - app["app_latency"]) > 0.01:
        raise ValueError(
            f"Critical path does not reconcile with application latency: {total}"
        )
    return totals | {"nodes": list(reversed(chain)), "total_s": total}


def read_run(path, graph=None):
    path = path.resolve()
    result = json.loads(path.read_text())
    directory = path.parent
    row = {
        "result_path": str(path),
        "result_sha256": digest(path),
        "identity": result["identity"],
        "launch": result["launch"],
        "qualifying": result["qualifying"],
        "exclusion_reasons": result.get("exclusion_reasons", []),
    }
    if not result["qualifying"]:
        return row
    if not result["correctness"]["passed"] or result.get("error"):
        raise ValueError(f"Qualifying result failed correctness checks: {path}")
    identity_fields = {
        key: value for key, value in result["identity"].items() if key != "key"
    }
    identity_hash = hashlib.sha256(
        json.dumps(identity_fields, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if identity_hash != result["identity"]["key"]:
        raise ValueError(f"Invalid experiment identity: {path}")
    sample_path = directory / "metrics.prom"
    samples = [
        {"name": sample.name, "value": sample.value, "labels": sample.labels}
        for family in text_string_to_metric_families(sample_path.read_text())
        for sample in family.samples
    ]
    application_files = list((directory / "app_results").glob("*.json"))
    if len(application_files) != 1:
        raise ValueError(f"Expected exactly one application result file: {path}")
    applications_path = application_files[0]
    applications = dict(
        sorted(
            json.loads(applications_path.read_text()).items(),
            key=lambda item: int(item[0]),
        )
    )
    app_latencies = [app["app_latency"] for app in applications.values()]
    performance = result["performance"]
    if set(applications) != {str(index) for index in range(24)} or not all(
        app["app_finished"] for app in applications.values()
    ):
        raise ValueError(f"Incomplete applications: {path}")
    if abs(mean(app_latencies) - performance["average_app_latency_s"]) > 1e-6:
        raise ValueError(f"Application latency mismatch: {path}")
    outcomes = (
        "physical_preempted",
        "reservation_preempted",
        "reservation_denied",
        "prefill_capacity_denied",
        "generation_capacity_denied",
        "generation_progress_deferred",
        "executed_tokens",
        "recomputed_tokens",
        "gpu_hit_tokens",
        "cpu_hit_tokens",
        "resume_gpu_hit_tokens",
        "resume_cpu_hit_tokens",
        "critical_admitted",
        "critical_wait_ge_60s",
        "critical_wait_ge_180s",
        "cache_affinity_admitted",
    )
    scheduling = {
        name: metric(samples, "vllm:tokencake_scheduling_total", outcome=name)
        for name in outcomes
    }
    decisions = {
        sample["labels"]["outcome"]: sample["value"]
        for sample in samples
        if sample["name"] == "vllm:tokencake_decision_total"
    }
    transfers = {}
    for direction in ("GPU_to_CPU", "CPU_to_GPU"):
        transfers[direction] = {
            "bytes": metric(
                samples, "vllm:kv_offload_total_bytes_total", transfer_type=direction
            ),
            "time_s": metric(
                samples, "vllm:kv_offload_total_time_total", transfer_type=direction
            ),
            "jobs": metric(
                samples, "vllm:kv_offload_size_count", transfer_type=direction
            ),
        }
    request_metrics = {}
    for name in (
        "request_queue_time_seconds",
        "request_prefill_time_seconds",
        "request_decode_time_seconds",
        "time_to_first_token_seconds",
        "e2e_request_latency_seconds",
        "request_prefill_kv_computed_tokens",
        "request_prompt_tokens",
    ):
        total, count = (
            metric(samples, f"vllm:{name}_sum"),
            metric(samples, f"vllm:{name}_count"),
        )
        request_metrics[name] = {
            "sum": total,
            "count": count,
            "mean": total / count if total is not None and count else None,
        }
    by_node = defaultdict(list)
    for app in applications.values():
        for name, node in app["request_info"].items():
            by_node[name].append(node)
    nodes = {}
    for name, values in sorted(by_node.items()):
        nodes[name] = {"type": values[0]["type"], "count": len(values)}
        for key in ("llm_latency", "tool_latency", "residual_latency", "latency"):
            numbers = [entry[key] for entry in values]
            nodes[name][key] = {
                "mean": mean(numbers),
                "p95": percentile(numbers, 0.95),
                "maximum": max(numbers),
            }
    config_path = directory / "resolved-server.json"
    config = json.loads(config_path.read_text())["vllm_config"]
    if config["speculative_config"] is not None:
        raise ValueError(f"Speculative decoding enabled: {path}")
    tokencake = config["additional_config"].get("tokencake")
    scheduling_enabled = tokencake is not None and tokencake.get("scheduling", {}).get(
        "enabled", True
    )
    if not scheduling_enabled:
        # The offload-only runtime exports zero-initialized scheduling counters
        # even though no scheduling observer records execution or admission.
        scheduling = dict.fromkeys(outcomes)
    row.update(
        performance=performance,
        application_ids=list(applications),
        application_latencies_s=app_latencies,
        application_arrival_offsets_s=[
            app["arrival_offset_s"] for app in applications.values()
        ],
        application_start_times_s=[
            app["app_start_time"] for app in applications.values()
        ],
        application_end_times_s=[app["app_end_time"] for app in applications.values()],
        token_counts=result["correctness"]["token_counts"],
        observed_workload_hash=result["correctness"]["observed_workload_hash"],
        output_text_compared=result["correctness"]["output_text_compared"],
        finished_preempted_count=result["correctness"]["finished_preempted_count"],
        terminal_status_counts=result["correctness"]["terminal_status_counts"],
        retries=result["attempt_diagnostics"]["retries"],
        prompt_halvings=result["attempt_diagnostics"]["prompt_halvings"],
        scheduling=scheduling,
        scheduling_observed=scheduling_enabled,
        decisions=decisions,
        transfers=transfers,
        native_preemptions=metric(samples, "vllm:num_preemptions_total"),
        iterations=metric(samples, "vllm:iteration_tokens_total_count"),
        critical_wait_max_s=metric(
            samples, "vllm:tokencake_critical_queue_wait_max_seconds"
        )
        if scheduling_enabled
        else None,
        critical_growth_wait_max_s=metric(
            samples, "vllm:tokencake_critical_growth_wait_max_seconds"
        )
        if scheduling_enabled
        else None,
        saved_blocks=metric(
            samples, "vllm:tokencake_saved_blocks_total", outcome="completed"
        ),
        request_metrics=request_metrics,
        first_prefill_sources={
            source: metric(samples, "vllm:prompt_tokens_by_source_total", source=source)
            for source in ("local_compute", "local_cache_hit", "external_kv_transfer")
        },
        node_metrics=nodes,
        settings={
            "model": config["model_config"]["model"],
            "dtype": config["model_config"]["dtype"],
            "max_model_len": config["model_config"]["max_model_len"],
            "speculative_config": config["speculative_config"],
            "cache_config": config["cache_config"],
            "scheduler_config": config["scheduler_config"],
            "additional_config": config["additional_config"],
        },
    )
    if graph is not None:
        campaign = next(
            parent for parent in path.parents if (parent / "frozen.json").exists()
        )
        launcher = json.loads((campaign / "launcher.json").read_text())
        if (
            launcher["materialized_helpers"][
                "workload-dataset.json"
                if "workload_dataset_sha256" in launcher
                else "agent/app/code_writer_paper_pressure.py"
            ]
            != graph["source_sha256"]
        ):
            raise ValueError(f"Graph metadata differs from frozen workload: {path}")
        row["critical_paths"] = {
            key: critical_path(app, graph) for key, app in applications.items()
        }
        latencies = [
            node["llm_latency"]
            for app in applications.values()
            for name, node in app["request_info"].items()
            if node.get("execution_kind") == "llm"
            and graph["nodes"][name]["metadata"].get("critical_path")
        ]
        row["annotated_critical_llm_latency_s"] = {
            "count": len(latencies),
            "mean": mean(latencies),
            "p95": percentile(latencies, 0.95),
            "maximum": max(latencies),
        }
    start, end = result["client_start_monotonic"], result["client_end_monotonic"]
    trace_path = directory / "metrics-timeseries.jsonl"
    if trace_path.exists():
        points = [json.loads(line) for line in trace_path.read_text().splitlines()]
        row["metrics_sampling"] = result.get("metrics_monitor")
        row["timeseries"] = {
            name: weighted_samples(
                points,
                start,
                end,
                lambda p, name=name: metric(p.get("metrics", []), f"vllm:{name}"),
            )
            for name in (
                "kv_cache_usage_perc",
                "num_requests_running",
                "num_requests_waiting",
            )
        }
    thermal_path = directory / "gpu-process-thermal.jsonl"
    if thermal_path.exists():
        points = [json.loads(line) for line in thermal_path.read_text().splitlines()]
        case = result["identity"]["case"]
        device = case.get("gpu_index")
        if device is None:
            device = int(case["mode"] in ("agent", "mooncake"))
        uuid = (
            (
                "GPU-dfe6761a-23f4-c34c-9c03-01da02020e5f"
                if device == 0
                else "GPU-ce81ab13-d8a0-a49d-c908-f876b8eb2087"
            )
            if device is not None
            else None
        )
        row["hardware_timeseries"] = {}
        for name in ("utilization.gpu", "temperature.gpu", "power.draw", "clocks.sm"):

            def value(point, name=name):
                values = [
                    entry.get(name)
                    for entry in point.get("thermal", [])
                    if entry.get("uuid") == uuid
                ]
                try:
                    return float(values[0]) if values else None
                except (ValueError, TypeError):
                    return None

            row["hardware_timeseries"][name] = weighted_samples(
                points, start, end, value
            )
    verified = []
    for artifact in (
        sample_path,
        applications_path,
        config_path,
        trace_path,
        thermal_path,
    ):
        if artifact.exists():
            relative = str(artifact.relative_to(directory))
            if digest(artifact) != result["artifacts"][relative]:
                raise ValueError(f"Changed artifact: {artifact}")
            verified.append(relative)
    row["verified_artifacts"] = verified
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--graph", type=Path)
    args = parser.parse_args()
    graph = json.loads(args.graph.read_text()) if args.graph else None
    rows = [
        read_run(path, graph)
        for root in args.roots
        for path in sorted(root.glob("cases/**/result.json"))
    ]
    if args.output:
        with args.output.open("x") as stream:
            json.dump(rows, stream, indent=2, sort_keys=True)
            stream.write("\n")
    for row in rows:
        print(
            json.dumps(
                {
                    "case": row["identity"]["case"],
                    "launch": row["launch"],
                    "qualifying": row["qualifying"],
                    "performance": row.get("performance"),
                    "exclusion_reasons": row["exclusion_reasons"],
                }
            )
        )


if __name__ == "__main__":
    main()
