# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import pytest

from tools.tokencake_experiments.campaign import Case
from tools.tokencake_experiments.driver import load_identity
from tools.tokencake_experiments.report import Identity, measurements, summarize


def identity(mode, qps=0.3):
    return Identity(Case(mode, qps), "code", "env", "launcher", "inputs", "config")


def result(case, duration, launch=0, qualifying=True):
    return {
        "identity": case.payload(),
        "launch": launch,
        "qualifying": qualifying,
        "performance": {"total_e2e_s": duration},
        "result_path": f"{case.case.name}/{launch}",
    }


def test_edited_identity_can_be_loaded_and_reported():
    case = identity("native")
    payload = case.payload()
    payload["artifact_sha256"] = "edited-code"
    assert load_identity(payload).artifact_sha256 == "edited-code"
    row = result(case, 100)
    row["identity"] = payload
    assert measurements(case, [row]) == [row]


def test_report_has_no_launch_cap_or_performance_acceptance_threshold():
    base, target = identity("native"), identity("offload-agent")
    rows = [result(base, 100)] + [result(target, 120, i) for i in range(5)]
    report = summarize([base, target], rows)
    assert report["cases"][1]["launches"] == 5
    assert report["comparisons"][0]["e2e_reduction_pct"] == pytest.approx(-20)
    assert report["comparisons"][0]["speedup"] == pytest.approx(100 / 120)
    assert "status" not in report and "requested_launches" not in report


def test_partial_report_keeps_missing_comparisons_empty():
    target = identity("agent")
    report = summarize([target], [result(target, 10)])
    assert report["cases"][0]["median_e2e_s"] == 10
    assert report["comparisons"][0]["speedup"] is None
    assert report["comparisons"][0]["native_median_e2e_s"] is None


@pytest.mark.parametrize("duration", [True, 0, -1, float("nan"), float("inf")])
def test_invalid_durations_are_omitted_without_blocking_report(duration):
    case = identity("native")
    report = summarize([case], [result(case, duration)])
    assert report["cases"][0]["launches"] == 1
    assert report["cases"][0]["median_e2e_s"] is None


def test_failed_launches_and_changed_code_do_not_hide_successful_results():
    original = identity("offload-agent")
    current = replace(original, artifact_sha256="changed")
    failed = result(original, 0, qualifying=False) | {
        "exclusion_reasons": ["client_failure"]
    }
    rows = [failed, result(current, 80, 1), result(current, 82, 2)]
    report = summarize([current], rows)
    assert report["cases"][0]["median_e2e_s"] == 81
    assert report["cases"][0]["launches"] == 3
    assert report["cases"][0]["excluded_launches"] == [
        {"launch": 0, "reasons": ["client_failure"]}
    ]
