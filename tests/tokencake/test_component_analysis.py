# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check comparability and time accounting before publishing component results."""

from copy import deepcopy

import pytest

from tools.tokencake_experiments.campaign import Case
from tools.tokencake_experiments.component_analysis import (
    critical_path,
    metric,
    weighted_samples,
)
from tools.tokencake_experiments.component_matrix import (
    LOADS,
    MODES,
    aggregate,
    group_runs,
)
from tools.tokencake_experiments.report import Identity


def test_weighted_measurement_clips_to_client_interval_and_preserves_gaps():
    points = [
        {"monotonic": 0, "value": 10},
        {"monotonic": 4, "value": 20},
        {"monotonic": 6, "value": None},
        {"monotonic": 8, "value": 1000},
    ]
    result = weighted_samples(points, 2, 8, lambda row: row["value"])
    assert result == {"mean": 15, "maximum": 20, "coverage_fraction": 4 / 6}


def test_long_sampling_gap_is_not_presented_as_measured_activity():
    points = [
        {"monotonic": 0, "value": 10},
        {"monotonic": 1, "value": 1000},
        {"monotonic": 21, "value": 30},
    ]
    result = weighted_samples(points, 0, 22, lambda row: row["value"])
    assert result == {"mean": 20, "maximum": 30, "coverage_fraction": 2 / 22}


def test_unsorted_or_duplicate_samples_are_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        weighted_samples([{"monotonic": 2}, {"monotonic": 2}], 0, 3, lambda _: 1)


def test_absent_metric_and_partial_aggregate_are_not_zero():
    samples = [{"name": "count", "value": 12, "labels": {"source": "gpu"}}]
    assert metric(samples, "count", source="cpu") is None
    assert aggregate([{"value": 12}, {"value": None}], lambda row: row["value"]) is None


def test_critical_path_uses_actual_join_dependencies_without_double_counting():
    dependencies = {
        "root": [],
        "fast": ["root"],
        "slow": ["root"],
        "join": ["fast", "slow"],
    }
    graph = {
        "nodes": {
            name: {"predecessors": predecessors}
            for name, predecessors in dependencies.items()
        }
    }
    timings = {"root": (0, 2), "fast": (2, 3), "slow": (2, 5), "join": (5.1, 6.1)}
    app = {
        "app_start_time": 0,
        "app_end_time": 6.2,
        "app_latency": 6.2,
        "request_info": {
            name: {
                "start_time": start,
                "end_time": end,
                "llm_latency": end - start if name != "fast" else 0,
                "tool_latency": end - start if name == "fast" else 0,
                "residual_latency": 0,
            }
            for name, (start, end) in timings.items()
        },
    }
    result = critical_path(app, graph)
    assert result["nodes"] == ["root", "slow", "join"]
    assert result["total_s"] == pytest.approx(6.2)
    assert result["llm_latency"] == 6
    assert result["tool_latency"] == 0
    assert result["orchestration_gap_s"] == pytest.approx(0.2)
    app["app_latency"] = 7.2
    with pytest.raises(ValueError, match="does not reconcile"):
        critical_path(app, graph)


@pytest.fixture
def matrix_rows():
    rows = []
    for mode in MODES:
        offload = mode in ("offload", "offload-agent")
        for qps in LOADS:
            identity = Identity(
                Case(mode, qps),
                "native" if mode == "native" else "target",
                "env",
                "launcher",
                "inputs",
                mode,
            ).payload()
            config = {
                "native": {},
                "agent": {"tokencake": {"offload": {"enabled": False}}},
                "offload": {"tokencake": {"scheduling": {"enabled": False}}},
                "offload-agent": {"tokencake": {}},
            }[mode]
            rows.append(
                {
                    "identity": identity,
                    "result_path": f"{mode}/{qps}/0",
                    "launch": 0,
                    "qualifying": True,
                    "observed_workload_hash": "observed",
                    "performance": {"completed_applications": 24},
                    "token_counts": {"generated": 155136},
                    "terminal_status_counts": {
                        "FINISHED_LENGTH_CAPPED": 648,
                        "FINISHED_LOCAL": 48,
                    },
                    "retries": 0,
                    "prompt_halvings": 0,
                    "finished_preempted_count": 0,
                    "settings": {
                        "model": "model",
                        "dtype": "bfloat16",
                        "max_model_len": 32768,
                        "speculative_config": None,
                        "scheduler_config": {"budget": 8192},
                        "cache_config": {
                            "num_gpu_blocks": 3050,
                            "kv_offloading_size": 100 if offload else None,
                            "kv_offloading_backend": "native",
                        },
                        "additional_config": deepcopy(config),
                    },
                    "application_arrival_offsets_s": [i / qps for i in range(24)],
                }
            )
    return rows


def test_complete_matrix_preserves_exclusions_and_valid_repeats(matrix_rows):
    repeat = deepcopy(matrix_rows[0])
    repeat.update(launch=2, result_path="repeat")
    excluded = {"qualifying": False, "result_path": "excluded", "launch": 1}
    groups = group_runs([*matrix_rows, repeat, excluded])
    assert len(groups) == 20
    assert len(groups["native", 0.05]) == 2


def test_incomplete_matrix_cannot_produce_full_report(matrix_rows):
    with pytest.raises(ValueError, match="all twenty"):
        group_runs(matrix_rows[:-1])


def test_duplicate_measurement_cannot_bias_median(matrix_rows):
    with pytest.raises(ValueError, match="Duplicate result"):
        group_runs([*matrix_rows, matrix_rows[0]])


@pytest.mark.parametrize(
    "field", ["workload_sha256", "launcher_sha256", "artifact_sha256"]
)
def test_mixed_frozen_artifacts_are_rejected(matrix_rows, field):
    matrix_rows[-1]["identity"][field] = "changed"
    with pytest.raises(ValueError, match=field):
        group_runs(matrix_rows)


@pytest.mark.parametrize(
    "change", ["tokens", "arrival", "cache", "component", "speculative", "retry"]
)
def test_work_or_configuration_drift_is_rejected(matrix_rows, change):
    row = matrix_rows[-1]
    if change == "tokens":
        row["token_counts"]["generated"] -= 1
    elif change == "arrival":
        row["application_arrival_offsets_s"][1] += 0.1
    elif change == "cache":
        row["settings"]["cache_config"]["num_gpu_blocks"] += 1
    elif change == "component":
        row["settings"]["additional_config"] = {
            "tokencake": {"scheduling": {"enabled": False}}
        }
    elif change == "speculative":
        row["settings"]["speculative_config"] = {"method": "unexpected"}
    else:
        row["retries"] = 1
    with pytest.raises(ValueError):
        group_runs(matrix_rows)
