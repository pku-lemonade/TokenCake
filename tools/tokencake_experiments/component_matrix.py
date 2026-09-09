# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit and tabulate the complete five-load, four-component experiment."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

MODES = ("native", "agent", "offload", "offload-agent")
LABELS = dict(zip(MODES, ("base", "agent", "offload", "agent_offload")))
LOADS = (0.05, 0.1, 0.2, 0.5, 1.0)


def group_runs(rows):
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        if row["result_path"] in seen:
            raise ValueError("Duplicate result supplied to component analysis")
        seen.add(row["result_path"])
        if row["qualifying"]:
            case = row["identity"]["case"]
            groups[case["mode"], case["qps"]].append(row)
    if set(groups) != {(mode, qps) for mode in MODES for qps in LOADS}:
        raise ValueError("Component analysis requires all twenty qualifying cases")
    valid = [row for group in groups.values() for row in group]
    for field in ("workload_sha256", "launcher_sha256"):
        if len({row["identity"][field] for row in valid}) != 1:
            raise ValueError(f"Different frozen {field} values in component matrix")
    if len({row["observed_workload_hash"] for row in valid}) != 1:
        raise ValueError("Observed workload contracts differ between launches")
    for mode in MODES:
        selected = [row for row in valid if row["identity"]["case"]["mode"] == mode]
        for field in ("artifact_sha256", "environment_sha256"):
            if len({row["identity"][field] for row in selected}) != 1:
                raise ValueError(f"Changed {field} within mode {mode}")
    target = [row for row in valid if row["identity"]["case"]["mode"] != "native"]
    if len({row["identity"]["artifact_sha256"] for row in target}) != 1:
        raise ValueError("Component modes must share one frozen target runtime")
    for key, group in groups.items():
        if len({row["identity"]["key"] for row in group}) != 1:
            raise ValueError(f"Mixed launch identities in {key}")
        launches = [row["launch"] for row in group]
        if len(set(launches)) != len(launches):
            raise ValueError(f"Duplicate qualifying launch number in {key}")
    scheduler_configs = []
    cache_configs = []
    model_configs = []
    for row in valid:
        if (
            row["performance"]["completed_applications"] != 24
            or row["token_counts"]["generated"] != 155136
            or row["terminal_status_counts"]
            != {"FINISHED_LENGTH_CAPPED": 648, "FINISHED_LOCAL": 48}
            or row["retries"]
            or row["prompt_halvings"]
            or row["finished_preempted_count"]
        ):
            raise ValueError(f"Incomplete or modified work: {row['result_path']}")
        settings = row["settings"]
        if settings["speculative_config"] is not None:
            raise ValueError("Speculative decoding is outside this experiment")
        scheduler_configs.append(settings["scheduler_config"])
        cache_configs.append(
            {
                key: value
                for key, value in settings["cache_config"].items()
                if key not in ("kv_offloading_size", "kv_offloading_backend")
            }
        )
        model_configs.append(
            {key: settings[key] for key in ("model", "dtype", "max_model_len")}
        )
        case = row["identity"]["case"]
        mode = case["mode"]
        offload = mode in ("offload", "offload-agent")
        if settings["cache_config"]["kv_offloading_size"] != (100 if offload else None):
            raise ValueError(f"Unexpected CPU KV capacity for {mode}")
        if offload and settings["cache_config"]["kv_offloading_backend"] != "native":
            raise ValueError(f"Unexpected CPU KV backend for {mode}")
        config = settings["additional_config"].get("tokencake")
        expected = {
            "native": None,
            "agent": {"offload": {"enabled": False}},
            "offload": {"scheduling": {"enabled": False}},
            "offload-agent": {},
        }[mode]
        if config != expected:
            raise ValueError(f"Unexpected component settings for {mode}: {config}")
    for name, configurations in (
        ("scheduler", scheduler_configs),
        ("cache", cache_configs),
        ("model", model_configs),
    ):
        if any(config != configurations[0] for config in configurations[1:]):
            raise ValueError(f"Resolved {name} settings differ across component modes")
    for qps in LOADS:
        offsets = [
            row["application_arrival_offsets_s"]
            for mode in MODES
            for row in groups[mode, qps]
        ]
        if any(value != offsets[0] for value in offsets[1:]):
            raise ValueError(f"Arrival trace mismatch at QPS {qps}")
    return groups


def aggregate(group, getter):
    values = [getter(row) for row in group]
    if any(value is None for value in values):
        return None
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Non-finite measurement")
    return median(values)


def summarize(groups):
    table = []
    for qps in LOADS:
        for mode in MODES:
            group = groups[mode, qps]
            row = {"qps": qps, "mode": LABELS[mode], "qualifying_launches": len(group)}
            row.update(
                {
                    name: aggregate(group, lambda r, name=name: r["performance"][name])
                    for name in group[0]["performance"]
                }
            )
            row.update(
                e2e_min_s=min(r["performance"]["total_e2e_s"] for r in group),
                e2e_max_s=max(r["performance"]["total_e2e_s"] for r in group),
                output_tokens_per_s=aggregate(
                    group,
                    lambda r: r["token_counts"]["generated"]
                    / r["performance"]["total_e2e_s"],
                ),
                native_preemptions=aggregate(group, lambda r: r["native_preemptions"]),
                critical_wait_max_s=aggregate(
                    group, lambda r: r["critical_wait_max_s"]
                ),
                critical_wait_observed_max_s=(
                    max(r["critical_wait_max_s"] for r in group)
                    if all(r["critical_wait_max_s"] is not None for r in group)
                    else None
                ),
            )
            for section, names in (
                ("scheduling", tuple(group[0]["scheduling"])),
                (
                    "first_prefill_sources",
                    ("local_compute", "local_cache_hit", "external_kv_transfer"),
                ),
            ):
                for name in names:
                    row[f"{section}.{name}"] = aggregate(
                        group, lambda r, name=name, section=section: r[section][name]
                    )
            for section, names in (
                (
                    "timeseries",
                    (
                        "kv_cache_usage_perc",
                        "num_requests_running",
                        "num_requests_waiting",
                    ),
                ),
                (
                    "hardware_timeseries",
                    ("utilization.gpu", "temperature.gpu", "power.draw", "clocks.sm"),
                ),
            ):
                for name in names:
                    row[f"{section}.{name}.mean"] = aggregate(
                        group,
                        lambda r, name=name, section=section: r[section][name]["mean"],
                    )
                    row[f"{section}.{name}.minimum_coverage"] = min(
                        r[section][name]["coverage_fraction"] for r in group
                    )
            for direction in ("GPU_to_CPU", "CPU_to_GPU"):
                for name in ("bytes", "time_s", "jobs"):
                    row[f"{direction}.{name}"] = aggregate(
                        group,
                        lambda r, name=name, direction=direction: r["transfers"][
                            direction
                        ][name],
                    )
            for name in (
                "llm_latency",
                "tool_latency",
                "residual_latency",
                "orchestration_gap_s",
            ):
                row[f"critical_path.{name}.mean"] = aggregate(
                    group,
                    lambda r, name=name: mean(
                        p[name] for p in r["critical_paths"].values()
                    ),
                )
            for name in ("mean", "p95", "maximum"):
                row[f"critical_node_llm.{name}"] = aggregate(
                    group,
                    lambda r, name=name: r["annotated_critical_llm_latency_s"][name],
                )
            for name in group[0]["request_metrics"]:
                row[f"request.{name}.mean"] = aggregate(
                    group, lambda r, name=name: r["request_metrics"][name]["mean"]
                )
            table.append(row)
    return table


def format_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_table(rows, columns):
    return "\n".join(
        [
            "| " + " | ".join(label for _, label in columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
            *(
                "| " + " | ".join(format_value(row[key]) for key, _ in columns) + " |"
                for row in rows
            ),
        ]
    )


def write_report(rows, destination):
    groups = group_runs(rows)
    table = summarize(groups)
    comparisons = []
    for qps in LOADS:
        by_mode = {row["mode"]: row for row in table if row["qps"] == qps}
        base, full = by_mode["base"], by_mode["agent_offload"]
        comparisons.append(
            {
                "qps": qps,
                **{
                    f"{mode}_e2e_reduction_pct": 100
                    * (1 - by_mode[mode]["total_e2e_s"] / base["total_e2e_s"])
                    for mode in ("agent", "offload", "agent_offload")
                },
                "full_p95_reduction_pct": 100
                * (1 - full["p95_app_latency_s"] / base["p95_app_latency_s"]),
                "interaction_s": by_mode["agent"]["total_e2e_s"]
                + by_mode["offload"]["total_e2e_s"]
                - base["total_e2e_s"]
                - full["total_e2e_s"],
            }
        )
    destination.mkdir(parents=True, exist_ok=False)
    payload = {
        "aggregation": (
            "Median of qualifying launches per metric, except explicitly "
            "reported observed maxima and minimum coverage; "
            "ranges are observed, not confidence intervals."
        ),
        "groups": table,
        "comparisons": comparisons,
        "excluded": [row for row in rows if not row["qualifying"]],
        "result_paths": [row["result_path"] for row in rows],
    }
    (destination / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    with (destination / "measurements.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    sections = [
        (
            "Latency and throughput",
            [
                ("qps", "QPS"),
                ("mode", "Mode"),
                ("qualifying_launches", "N"),
                ("total_e2e_s", "E2E (s)"),
                ("average_app_latency_s", "Mean (s)"),
                ("p50_app_latency_s", "P50 (s)"),
                ("p90_app_latency_s", "P90 (s)"),
                ("p95_app_latency_s", "P95 (s)"),
                ("p99_app_latency_s", "P99 (s)"),
                ("max_app_latency_s", "Max (s)"),
                ("output_tokens_per_s", "Output token/s"),
            ],
        ),
        (
            "Scheduling",
            [
                ("qps", "QPS"),
                ("mode", "Mode"),
                ("native_preemptions", "Native preemptions"),
                *(
                    (f"scheduling.{key}", label)
                    for key, label in (
                        ("physical_preempted", "Physical"),
                        ("reservation_preempted", "Reservation"),
                        ("reservation_denied", "Reservation denies"),
                        ("generation_capacity_denied", "Generation denies"),
                        ("prefill_capacity_denied", "Prefill denies"),
                        ("recomputed_tokens", "Recomputed tokens"),
                        ("resume_gpu_hit_tokens", "Resume GPU tokens"),
                        ("resume_cpu_hit_tokens", "Resume CPU tokens"),
                    )
                ),
            ],
        ),
        (
            "Important requests",
            [
                ("qps", "QPS"),
                ("mode", "Mode"),
                ("critical_wait_max_s", "Median max critical admission wait (s)"),
                (
                    "critical_wait_observed_max_s",
                    "Observed max critical admission wait (s)",
                ),
                ("scheduling.critical_wait_ge_60s", "Critical waits >= 60 s"),
                ("scheduling.critical_wait_ge_180s", "Critical waits >= 180 s"),
                ("critical_node_llm.mean", "Critical node mean LLM (s)"),
                ("critical_node_llm.p95", "Critical node P95 LLM (s)"),
            ],
        ),
        (
            "Critical path decomposition",
            [
                ("qps", "QPS"),
                ("mode", "Mode"),
                *(
                    (f"critical_path.{name}.mean", label)
                    for name, label in (
                        ("llm_latency", "LLM (s)"),
                        ("tool_latency", "Tool (s)"),
                        ("residual_latency", "Residual (s)"),
                        ("orchestration_gap_s", "Orchestration gap (s)"),
                    )
                ),
            ],
        ),
    ]
    body = [
        "# Complete Component Measurements",
        "",
        payload["aggregation"],
        "",
        "N/A means that the runtime did not expose the measurement. "
        "It does not mean zero.",
        "",
        "## Relative Effects",
        "",
        markdown_table(comparisons, [(key, key) for key in comparisons[0]]),
    ]
    for title, columns in sections:
        body.extend(["", f"## {title}", "", markdown_table(table, columns)])
    (destination / "tables.md").write_text("\n".join(body) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = write_report(json.loads(args.data.read_text()), args.output)
    print(
        json.dumps(
            {
                "cases": len(result["groups"]),
                "excluded": len(result["excluded"]),
                "comparisons": result["comparisons"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
