# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import pytest

from tools.tokencake_experiments.campaign import Case, initial_queues
from tools.tokencake_experiments.driver import load_identity
from tools.tokencake_experiments.report import (
    Identity,
    compare,
    content_hash,
    evaluate,
    measurements,
)


def identity(mode, qps=1.0, phase="phase-1"):
    return Identity(
        Case(mode, qps, phase), "code", "env", "launcher", "inputs", "config"
    )


def test_default_gpu_preserves_historical_identity_hash():
    fields = {
        "case": {"mode": "agent", "qps": 1.0, "phase": "phase-1"},
        "artifact_sha256": "code",
        "environment_sha256": "env",
        "launcher_sha256": "launcher",
        "workload_sha256": "inputs",
        "config_sha256": "config",
    }
    historical = fields | {"key": content_hash(fields)}
    assert load_identity(historical).payload() == historical
    explicit = replace(identity("agent"), case=Case("agent", 1.0, gpu_index=0))
    assert explicit.key != historical["key"]
    assert load_identity(explicit.payload()) == explicit


def test_twenty_five_percent_gate_is_separate_from_historical_gate():
    target, reference = identity("offload-agent"), identity("native")
    rows = [
        result(case, seconds, launch)
        for launch in range(3)
        for case, seconds in ((target, 80), (reference, 100))
    ]
    historical = compare(target, reference, rows)
    current = compare(target, reference, rows, native_improvement=0.25)
    assert historical["status"] == "pass"
    assert current["status"] == "fail"
    assert current["maximum_ratio"] == 0.75
    assert current["gate_id"] != historical["gate_id"]


def test_single_qps_diagnostic_cannot_pass_the_three_qps_acceptance():
    identities = [identity(mode) for mode in ("native", "agent", "offload-agent")]
    rows = [
        result(case, 70 if case.case.mode == "offload-agent" else 100)
        for case in identities
    ]
    report = evaluate(identities, rows, include_old=False, native_improvement=0.25)
    assert len(report["gates"]) == 2
    assert all(gate["status"] == "pass" for gate in report["gates"])
    assert not report["passed"]
    assert not report["applicable_gates_passed"]
    assert report["requested_launches"] == {}
    assert {
        (item["qps"], item["reference_mode"])
        for item in report["unprepared_comparisons"]
    } == {(qps, mode) for qps in (0.5, 0.1) for mode in ("native", "agent")}


def result(case, seconds, launch=0, *, qualifying=True, preempted=0):
    return {
        "identity": case.payload(),
        "launch": launch,
        "qualifying": qualifying,
        "performance": {"total_e2e_s": seconds},
        "correctness": {"finished_preempted_count": preempted},
        "result_path": f"{case.key}/{launch}/result.json",
    }


def test_expanded_gate_report_checks_both_new_qps_points():
    qps_values = (1.0, 0.5, 0.2, 0.1, 0.05)
    identities = [
        identity(mode, qps)
        for mode in ("native", "agent", "offload-agent")
        for qps in qps_values
    ]
    rows = [
        result(case, 70 if case.case.mode == "offload-agent" else 100)
        for case in identities
    ]
    report = evaluate(
        identities,
        rows,
        include_old=False,
        native_improvement=0.25,
        qps_values=qps_values,
    )
    assert report["passed"] and len(report["gates"]) == 10
    assert {gate["qps"] for gate in report["gates"]} == set(qps_values)


@pytest.mark.parametrize(
    "mode,boundary", [("native", 0.9), ("agent", 1.05), ("old-offload-agent", 1.0)]
)
@pytest.mark.parametrize("offset", [-0.031, -0.029, 0, 0.001, 0.1])
def test_strict_gates_and_gray_band(mode, boundary, offset):
    target, reference = identity("offload-agent"), identity(mode)
    rows = [result(target, 100 * (boundary + offset)), result(reference, 100)]
    report = compare(target, reference, rows)
    if offset < -0.03:
        assert report["status"] == "pass"
        assert report["requested_launches"] == {}
        return
    assert report["requested_launches"] == {target.key: 2, reference.key: 2}
    rows += [
        result(case, seconds, launch)
        for launch in (1, 2)
        for case, seconds in ((target, 100 * (boundary + offset)), (reference, 100))
    ]
    report = compare(target, reference, rows)
    assert report["status"] == ("pass" if offset <= 0 else "fail")
    assert report["requested_launches"] == {}


def test_medians_exclude_invalid_launches_without_replacing_them():
    target, reference = identity("offload-agent"), identity("native")
    rows = [
        result(target, time, i, qualifying=i != 1)
        for i, time in enumerate((100, 1000, 80))
    ]
    rows += [result(reference, time, i) for i, time in enumerate((100, 102, 1000))]
    report = compare(target, reference, rows)
    assert report["target_median_s"] == 90
    assert report["reference_median_s"] == 102
    assert report["status"] == "pass"
    assert len(report["gate_inputs"][target.key]) == 2
    assert report["requested_launches"] == {}


def test_affected_pairs_reuse_shared_case_and_do_not_repeat_mooncake():
    identities = [
        identity(case.mode, case.qps)
        for queues in initial_queues().values()
        for queue in queues
        for case in queue
    ]
    durations = {
        "native": 100,
        "agent": 110,
        "offload-agent": 80,
        "old-offload-agent": 120,
        "mooncake": 1,
    }
    rows = [
        result(
            case,
            91
            if case.case.mode == "offload-agent" and case.case.qps == 1
            else durations[case.case.mode],
        )
        for case in identities
    ]
    report = evaluate(identities, rows)
    target = next(case for case in identities if case.case == Case("offload-agent", 1))
    native = next(case for case in identities if case.case == Case("native", 1))
    assert report["requested_launches"] == {target.key: 2, native.key: 2}
    for case in (target, native):
        rows += [result(case, 89 if case == target else 100, i) for i in (1, 2)]
    old = next(case for case in identities if case.case == Case("old-offload-agent", 1))
    next(row for row in rows if row["identity"]["key"] == old.key)["performance"][
        "total_e2e_s"
    ] = 90
    report = evaluate(identities, rows)
    assert report["requested_launches"] == {old.key: 2}
    assert all(
        item["launches"] == 1 and not item["gating"]
        for item in report["mooncake_speed_reference"]
    )


def test_contamination_budget_and_old_work_equivalence():
    target, reference = identity("offload-agent"), identity("native")
    rows = [result(target, 10, i, qualifying=False) for i in range(3)]
    report = compare(target, reference, rows)
    assert report["reason"] == "exhausted_nonqualifying_budget"
    assert report["requested_launches"] == {}
    with pytest.raises(ValueError, match="launch ledger"):
        measurements(target, rows + [result(target, 10, 3)])
    old = identity("old-offload-agent")
    report = compare(target, old, [result(target, 80), result(old, 50, preempted=1)])
    assert report["status"] == "work_inequivalent_reference"
    assert report["requested_launches"] == {}
    assert report["gate_inputs"][old.key] == []
    mooncake = identity("mooncake")
    assert len(measurements(mooncake, [result(mooncake, 0, qualifying=False)])) == 1
    with pytest.raises(ValueError, match="launch ledger"):
        measurements(
            mooncake, [result(mooncake, 0, qualifying=False), result(mooncake, 10, 1)]
        )


def test_phase_two_requires_reference_proof_and_has_separate_budget():
    phase1, phase2, reference = (
        identity("offload-agent"),
        identity("offload-agent", phase="phase-2"),
        identity("native"),
    )
    phase2 = replace(phase2, artifact_sha256="new-code")
    rows = [result(phase1, 1000, i) for i in range(3)] + [result(reference, 100)]
    assert (
        compare(phase2, reference, rows)["reason"] == "reference_equivalence_required"
    )
    kwargs = {"reference_equivalence": {reference.key: "proof.json#native-unchanged"}}
    assert compare(phase2, reference, rows, **kwargs)["requested_launches"] == {
        phase2.key: 1
    }
    rows.append(result(phase2, 80))
    report = compare(phase2, reference, rows, **kwargs)
    assert report["status"] == "pass" and report["target_median_s"] == 80
    assert report["launches"][phase2.key] == 1
    with pytest.raises(ValueError, match="one implementation identity"):
        evaluate([phase1, phase2], rows)


@pytest.mark.parametrize("seconds", [True, 0, -1, float("nan"), float("inf")])
def test_invalid_qualifying_duration_is_rejected(seconds):
    case = identity("native")
    with pytest.raises(ValueError, match="positive finite"):
        measurements(case, [result(case, seconds)])


def test_forged_identity_and_input_mismatch_are_rejected():
    case = identity("native")
    row = result(case, 100)
    row["identity"]["artifact_sha256"] = "different"
    with pytest.raises(ValueError, match="recorded hash"):
        measurements(case, [row])
    with pytest.raises(ValueError, match="different frozen inputs"):
        compare(replace(identity("offload-agent"), workload_sha256="changed"), case, [])


def test_correctness_repair_keeps_excluded_launch_charges():
    original = identity("offload-agent")
    repaired = replace(original, artifact_sha256="correctness-fix")
    excluded = result(original, 0, qualifying=False) | {"budget_identity": repaired.key}
    rows = [excluded, result(repaired, 80, 1), result(repaired, 82, 2)]
    assert len(measurements(repaired, rows)) == 3
    reference = identity("native")
    report = compare(repaired, reference, rows + [result(reference, 100)])
    assert report["target_median_s"] == 81
    assert report["launches"][repaired.key] == 3
    assert report["gate_inputs"][repaired.key] == [
        row["result_path"] for row in rows[1:]
    ]
    with pytest.raises(ValueError, match="launch ledger"):
        measurements(repaired, rows + [result(repaired, 83, 3)])
    with pytest.raises(ValueError, match="Only exclusions"):
        measurements(repaired, [excluded | {"qualifying": True}])
    with pytest.raises(ValueError, match="different experiment contracts"):
        measurements(
            replace(repaired, config_sha256="different"),
            [
                excluded
                | {"budget_identity": replace(repaired, config_sha256="different").key}
            ],
        )
