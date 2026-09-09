# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measurement boundaries, engine counter resets, and incomplete sampling."""

import json

import pytest

from tools.tokencake_experiments.agent_bench.report import observations
from tools.tokencake_experiments.agent_bench.throughput import (
    GENERATION_COUNTER,
    output_throughput,
    read_metric_samples,
    shared_measurement_start,
    throughput_comparisons,
)


def sample(timestamp, engine_values):
    return {
        "timestamp": timestamp,
        "metrics": [
            {
                "name": GENERATION_COUNTER,
                "labels": {"model_name": "fixture", "engine": str(i)},
                "value": value,
            }
            for i, value in enumerate(engine_values)
        ]
        + [
            {
                "name": "vllm:prompt_tokens_total",
                "labels": {"model_name": "fixture"},
                "value": 999999,
            },
            {
                "name": GENERATION_COUNTER,
                "labels": {"model_name": "other", "engine": "0"},
                "value": 999999,
            },
        ],
    }


def baseline(directory, values):
    (directory / "metrics-start.prom").write_text(
        "# TYPE vllm:generation_tokens_total counter\n"
        + "\n".join(
            f'vllm:generation_tokens_total{{model_name="fixture",engine="{i}"}} {v}'
            for i, v in enumerate(values)
        )
        + "\n"
    )


def analyze(directory, execution, tokens=100):
    samples, issues = read_metric_samples(directory)
    return output_throughput(
        directory, execution, tokens, "fixture", 100, samples, issues
    )


def test_throughput_uses_window_counter_delta_and_one_wall_clock(tmp_path):
    baseline(tmp_path, [5, 10])
    rows = [
        sample(100, [5, 10]),
        sample(1899, [80, 120]),
        sample(1901, [100, 150]),
        sample(2499, [150, 250]),
        sample(2501, [10000, 10000]),
    ]
    (tmp_path / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    # The shutdown snapshot must never replace the cutoff sample.
    (tmp_path / "metrics.prom").write_text(
        'vllm:generation_tokens_total{model_name="fixture",engine="0"} 999999\n'
    )
    execution = {"started_at": 100, "ended_at": 2500, "wall_s": 2400}
    result = analyze(tmp_path, execution, tokens=350)
    short, full = (result["windows"][key] for key in ["first_30_minutes", "full_run"])
    assert short["output_tokens_delta"] == 185
    assert short["tokens_per_second_per_gpu"] == pytest.approx(185 / 1800)
    assert short["sample_lag_s"] == 1
    assert full["output_tokens_delta"] == 385
    assert full["tokens_per_second_per_gpu"] == pytest.approx(385 / 2400)
    assert result["client_complete_responses"][
        "tokens_per_second_per_gpu"
    ] == pytest.approx(350 / 2400)
    assert result["sampling_audit"]["counter_resets"] == []
    obs = observations(tmp_path, execution)
    assert obs["counter_window"]["last_sample_timestamp"] == 2499
    assert [
        r["delta"]
        for r in obs["counter_deltas"]
        if r["name"] == GENERATION_COUNTER and r["labels"]["model_name"] == "fixture"
    ] == [145, 240]


def test_engine_reset_cannot_hide_behind_an_increasing_total(tmp_path):
    baseline(tmp_path, [0, 0])
    rows = [sample(100, [0, 0]), sample(1800, [100, 200]), sample(1899, [90, 400])]
    (tmp_path / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)))
    result = analyze(tmp_path, {"started_at": 100, "ended_at": 1900, "wall_s": 1800})
    assert len(result["sampling_audit"]["counter_resets"]) == 1
    for window in result["windows"].values():
        assert window["tokens_per_second_per_gpu"] is None
        assert "counter_reset_in_window" in window["reasons"]


def test_missing_samples_are_visible_and_early_finish_does_not_invent_30_minutes(
    tmp_path,
):
    baseline(tmp_path, [0])
    rows = [
        json.dumps(sample(100, [0])),
        '{"timestamp":',
        json.dumps(sample(900, [100])),
        json.dumps(sample(999, [])),
    ]
    (tmp_path / "metrics.jsonl").write_text("\n".join(rows))
    result = analyze(tmp_path, {"started_at": 100, "ended_at": 1000, "wall_s": 900})
    assert result["windows"]["full_run"]["last_valid_sample"]["timestamp"] == 900
    assert result["windows"]["full_run"]["sample_lag_s"] == 100
    assert result["windows"]["first_30_minutes"]["tokens_per_second_per_gpu"] is None
    assert {i["kind"] for i in result["sampling_audit"]["issues"]} == {
        "invalid_sample",
        "counter_series_missing_or_changed",
    }
    assert result["sampling_audit"]["gaps_over_4s"] == 1


def test_shared_window_origin_comes_from_matching_budget_record(tmp_path):
    ledger = tmp_path / "budget.jsonl"
    config = {
        "output": "/fixture/run",
        "modes": ["agent_offload", "base"],
        "manifest_sha256": "frozen",
        "budget_ledger": str(ledger),
    }
    records = [
        {"event": "budget", "manifest_sha256": "frozen"},
        {
            "event": "start",
            "label": "/different/run:parallel:agent_offload,base",
            "time": 1,
        },
        {
            "event": "start",
            "label": "/fixture/run:parallel:agent_offload,base",
            "time": 100,
        },
    ]
    ledger.write_text("\n".join(map(json.dumps, records)))
    assert shared_measurement_start(config)["timestamp"] == 100
    assert shared_measurement_start(config)["line"] == 3
    config["manifest_sha256"] = "changed"
    assert shared_measurement_start(config)["timestamp"] is None


def test_relative_throughput_handles_zero_and_unavailable_without_speedup(tmp_path):
    def data(rate):
        row = {"tokens_per_second_per_gpu": rate}
        return {
            "output_throughput": {
                "windows": {"first_30_minutes": row, "full_run": row},
                "client_complete_responses": row,
            }
        }

    result = throughput_comparisons({"base": data(40), "agent_offload": data(50)})
    assert result["agent_offload"]["full_run"]["relative_change_percent"] == 25
    for left, right in [(0, 50), (40, None), (None, 50)]:
        result = throughput_comparisons(
            {"base": data(left), "agent_offload": data(right)}
        )
        assert result["agent_offload"]["full_run"]["relative_change_percent"] is None
