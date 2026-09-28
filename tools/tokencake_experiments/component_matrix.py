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
            continue
        seen.add(row["result_path"])
        if row["qualifying"]:
            case = row["identity"]["case"]
            groups[case["mode"], case["qps"]].append(row)
    return groups


def aggregate(group, getter):
    values = []
    for row in group:
        try:
            values.append(getter(row))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            values.append(None)
    if not values:
        return None
    if any(value is None for value in values):
        return None
    if any(not math.isfinite(value) for value in values):
        return None
    return median(values)


def summarize(groups):
    table = []
    for qps in sorted({qps for _, qps in groups}):
        for mode in MODES:
            group = groups.get((mode, qps), [])
            if not group:
                continue
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
                    if all(r.get("critical_wait_max_s") is not None for r in group)
                    else None
                ),
            )
            for section, names in (
                ("scheduling", tuple(group[0].get("scheduling", {}))),
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
                        r.get(section, {}).get(name, {}).get("coverage_fraction", 0)
                        for r in group
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
            for name in group[0].get("request_metrics", {}):
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
                "| "
                + " | ".join(format_value(row.get(key)) for key, _ in columns)
                + " |"
                for row in rows
            ),
        ]
    )


def write_report(rows, destination):
    groups = group_runs(rows)
    table = summarize(groups)
    comparisons = []
    for qps in sorted({row["qps"] for row in table}):
        by_mode = {row["mode"]: row for row in table if row["qps"] == qps}
        base = by_mode.get("base")
        if base is None:
            continue
        comparison = {"qps": qps}
        for mode in ("agent", "offload", "agent_offload"):
            current = by_mode.get(mode)
            comparison[f"{mode}_e2e_reduction_pct"] = (
                100 * (1 - current["total_e2e_s"] / base["total_e2e_s"])
                if current
                else None
            )
        full = by_mode.get("agent_offload")
        comparison["full_p95_reduction_pct"] = (
            100 * (1 - full["p95_app_latency_s"] / base["p95_app_latency_s"])
            if full and base.get("p95_app_latency_s")
            else None
        )
        comparison["interaction_s"] = (
            by_mode["agent"]["total_e2e_s"]
            + by_mode["offload"]["total_e2e_s"]
            - base["total_e2e_s"]
            - full["total_e2e_s"]
            if full and "agent" in by_mode and "offload" in by_mode
            else None
        )
        comparisons.append(comparison)
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
        writer = csv.DictWriter(
            stream,
            fieldnames=sorted({key for row in table for key in row}),
            lineterminator="\n",
        )
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
        "# Component Measurements",
        "",
        payload["aggregation"],
        "",
        "N/A means that the runtime did not expose the measurement. "
        "It does not mean zero.",
        "",
        "## Relative Effects",
        "",
        markdown_table(
            comparisons, [(key, key) for key in comparisons[0]] if comparisons else []
        ),
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
