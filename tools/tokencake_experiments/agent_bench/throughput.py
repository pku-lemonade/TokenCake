# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Output throughput from cumulative counters bounded by measurement windows."""

import json
import math
from pathlib import Path

from tools.tokencake_experiments.runtime import metric_values

GENERATION_COUNTER = "vllm:generation_tokens_total"


def read_metric_samples(service: Path) -> tuple[list[dict], list[dict]]:
    samples, issues = [], []
    path = service / "metrics.jsonl"
    if not path.exists():
        return [], [{"kind": "metrics_file_missing"}]
    with path.open() as stream:
        previous = None
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                timestamp = row["timestamp"]
                if not isinstance(timestamp, (float, int)) or not math.isfinite(
                    timestamp
                ):
                    raise ValueError("Invalid timestamp")
                if not isinstance(row.get("metrics", []), list):
                    raise ValueError("Invalid metric list")
                for metric in row.get("metrics", []):
                    if (
                        not isinstance(metric, dict)
                        or not isinstance(metric.get("name"), str)
                        or not isinstance(metric.get("labels"), dict)
                        or not isinstance(metric.get("value"), (float, int))
                    ):
                        raise ValueError("Invalid metric record")
            except (ValueError, TypeError, KeyError) as exc:
                issues.append(
                    {"kind": "invalid_sample", "line": number, "error": str(exc)}
                )
                continue
            if previous is not None and timestamp <= previous:
                issues.append(
                    {
                        "kind": "timestamp_not_increasing",
                        "line": number,
                        "timestamp": timestamp,
                    }
                )
            previous = timestamp
            if row.get("error"):
                issues.append(
                    {
                        "kind": "scrape_error",
                        "line": number,
                        "timestamp": timestamp,
                        "error": row["error"],
                    }
                )
            samples.append(row | {"line": number})
    return samples, issues


def shared_measurement_start(config: dict) -> dict:
    path = Path(config.get("budget_ledger", ""))
    if not path.is_file():
        return {"timestamp": None, "reason": "budget_ledger_missing"}
    label = f"{config.get('output')}:parallel:{','.join(config['modes'])}"
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    if not records or records[0].get("manifest_sha256") != config["manifest_sha256"]:
        return {"timestamp": None, "reason": "budget_ledger_identity_mismatch"}
    starts = [
        (index, row)
        for index, row in enumerate(records, 1)
        if row.get("event") == "start" and row.get("label") == label
    ]
    if len(starts) != 1:
        return {"timestamp": None, "reason": "unique_parallel_start_missing"}
    number, record = starts[0]
    return {
        "timestamp": record["time"],
        "source": str(path),
        "line": number,
        "label": label,
    }


def _series(metrics: list[dict], model_name: str | None) -> dict:
    values = {}
    for row in metrics:
        if not isinstance(row, dict):
            raise ValueError("Invalid metric record")
        if row.get("name") != GENERATION_COUNTER:
            continue
        labels = row["labels"]
        if labels.get("model_name") != model_name:
            continue
        key = tuple(sorted(labels.items()))
        value = row["value"]
        if (
            key in values
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("Duplicate series or invalid generation counter")
        values[key] = value
    return values


def output_throughput(
    service: Path,
    execution: dict,
    client_output_tokens: int,
    model_name: str | None,
    common_start: float | None,
    samples: list[dict],
    sample_issues: list[dict],
) -> dict:
    issues = list(sample_issues)
    baseline = {}
    path = service / "metrics-start.prom"
    try:
        baseline = _series(metric_values(path.read_text()), model_name)
    except (OSError, ValueError) as exc:
        issues.append({"kind": "baseline_error", "error": str(exc)})
    if not baseline:
        issues.append({"kind": "baseline_counter_missing"})
    valid, resets = [], []
    previous = baseline
    for sample in samples:
        marker = {"line": sample["line"], "timestamp": sample["timestamp"]}
        try:
            current = _series(sample.get("metrics", []), model_name)
        except (ValueError, TypeError, KeyError) as exc:
            issues.append(marker | {"kind": "invalid_counter", "error": str(exc)})
            continue
        if not baseline or current.keys() != baseline.keys():
            issues.append(marker | {"kind": "counter_series_missing_or_changed"})
            continue
        for key, value in current.items():
            if value < previous[key]:
                resets.append(
                    marker
                    | {"labels": dict(key), "previous": previous[key], "value": value}
                )
        previous = current
        valid.append(marker | {"counter": sum(current.values())})
    initial = sum(baseline.values()) if baseline else None
    gaps = [b["timestamp"] - a["timestamp"] for a, b in zip(valid, valid[1:])]
    wall = execution.get("wall_s")
    end = execution.get("ended_at")

    def window(start, cutoff, denominator, *, require_full_window=False):
        reasons = []
        if initial is None:
            reasons.append("baseline_unavailable")
        if start is None or cutoff is None or not denominator or denominator <= 0:
            reasons.append("window_timing_unavailable")
        elif require_full_window and (end is None or end < cutoff):
            reasons.append("run_ended_before_30_minute_window")
        candidates = [
            r for r in valid if cutoff is not None and r["timestamp"] <= cutoff
        ]
        latest = max(candidates, key=lambda r: r["timestamp"]) if candidates else None
        if latest is None or (start is not None and latest["timestamp"] < start):
            reasons.append("window_sample_unavailable")
        if cutoff is not None and any(r["timestamp"] <= cutoff for r in resets):
            reasons.append("counter_reset_in_window")
        if any(r["kind"] == "timestamp_not_increasing" for r in issues):
            reasons.append("sample_timestamp_order_invalid")
        tokens = latest["counter"] - initial if latest and initial is not None else None
        rate = tokens / denominator if not reasons else None
        inside = [
            r
            for r in valid
            if start is not None
            and cutoff is not None
            and start <= r["timestamp"] <= cutoff
        ]
        intervals = [
            b["timestamp"] - a["timestamp"] for a, b in zip(inside, inside[1:])
        ]
        return {
            "status": "unavailable" if reasons else "approximate",
            "reasons": reasons,
            "started_at": start,
            "cutoff": cutoff,
            "denominator_s": denominator,
            "last_valid_sample": latest,
            "sample_lag_s": cutoff - latest["timestamp"] if latest else None,
            "output_tokens_delta": tokens,
            "tokens_per_second_per_gpu": rate,
            "valid_samples_in_window": len(inside),
            "max_sample_interval_s": max(intervals) if intervals else None,
        }

    return {
        "metric": GENERATION_COUNTER,
        "model_name": model_name,
        "initial_counter": initial,
        "engine_series": [dict(key) for key in baseline],
        "windows": {
            "first_30_minutes": window(
                common_start,
                common_start + 1800 if common_start is not None else None,
                1800,
                require_full_window=True,
            ),
            "full_run": window(execution.get("started_at"), end, wall),
        },
        "client_complete_responses": {
            "output_tokens": client_output_tokens,
            "wall_s": wall,
            "tokens_per_second_per_gpu": client_output_tokens / wall
            if wall and wall > 0
            else None,
        },
        "sampling_audit": {
            "parsed_samples": len(samples),
            "valid_counter_samples": len(valid),
            "issues": issues,
            "counter_resets": resets,
            "max_interval_s": max(gaps) if gaps else None,
            "gaps_over_4s": sum(gap > 4 for gap in gaps),
            "gap_interpretation": (
                "Intervals above 4 seconds flag possible gaps "
                "for the configured 2-second sampling."
            ),
        },
        "interpretation": [
            "Sum engine counters at one timestamp, subtract baseline, "
            "divide by one wall clock. Each mode uses one GPU.",
            "Use the last valid sample at or before the cutoff, never shutdown "
            "metrics.prom. Sampling makes this approximate.",
            "Keep queueing, prefill, decode, real tools, waits and failures in the "
            "wall clock; exclude model loading and post-run idle.",
            "Client output counts only complete responses with usage. Server "
            "counters can include aborted or timed-out requests.",
            "Generated tokens can belong to incorrectly solved tasks; "
            "output throughput is not correct-task throughput.",
            "Report equal 30-minute and full-run windows together. "
            "Full-run trajectories and durations can differ.",
        ],
    }


def throughput_comparisons(modes: dict, reference: str = "base") -> dict:
    if reference not in modes:
        return {}
    comparisons = {}

    def rate(item, name):
        value = item["output_throughput"]
        row = (
            value[name]
            if name == "client_complete_responses"
            else value["windows"][name]
        )
        return row["tokens_per_second_per_gpu"]

    for mode, data in modes.items():
        if mode == reference:
            continue
        comparison = {}
        for name in ("first_30_minutes", "full_run", "client_complete_responses"):
            left, right = rate(modes[reference], name), rate(data, name)
            comparison[name] = {
                "reference": reference,
                "relative_change_percent": (right / left - 1) * 100
                if left and right is not None
                else None,
            }
        comparisons[mode] = comparison
    return comparisons
