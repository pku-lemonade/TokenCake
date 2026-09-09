# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness-qualified medians and the accepted per-comparison launch budget."""

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

from tools.tokencake_experiments.campaign import QPS, Case, Mode

BOUNDARIES: dict[Mode, float] = {
    "native": 0.90,
    "agent": 1.05,
    "old-offload-agent": 1.0,
}
MAX_LAUNCHES = 3
GRAY_BAND = 0.03


def content_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Identity:
    case: Case
    artifact_sha256: str
    environment_sha256: str
    launcher_sha256: str
    workload_sha256: str
    config_sha256: str

    @property
    def key(self) -> str:
        return self.payload()["key"]

    def payload(self) -> dict:
        fields = asdict(self)
        if self.case.gpu_index is None:
            fields["case"].pop("gpu_index")
        return fields | {"key": content_hash(fields)}


def measurements(identity: Identity, results: list[dict]) -> list[dict]:
    rows = sorted(
        (
            r
            for r in results
            if r.get("budget_identity", r["identity"]["key"]) == identity.key
        ),
        key=lambda r: r["launch"],
    )
    numbers = [row["launch"] for row in rows]
    limit = 1 if identity.case.mode == "mooncake" else MAX_LAUNCHES
    if numbers != list(range(len(rows))) or len(rows) > limit:
        raise ValueError(f"Invalid launch ledger for {identity.case.name}: {numbers}")
    for row in rows:
        actual = row["identity"]
        if (
            content_hash({k: v for k, v in actual.items() if k != "key"})
            != actual["key"]
        ):
            raise ValueError("Result identity does not match its recorded hash")
        if "budget_identity" in row:
            if row.get("qualifying") or actual["case"] != identity.payload()["case"]:
                raise ValueError(
                    "Only exclusions from the same phase/mode/QPS "
                    "may carry a launch charge"
                )
            if any(
                actual[key] != identity.payload()[key]
                for key in (
                    "environment_sha256",
                    "launcher_sha256",
                    "workload_sha256",
                    "config_sha256",
                )
            ):
                raise ValueError(
                    "An exclusion cannot transfer between different "
                    "experiment contracts"
                )
        elif actual != identity.payload():
            raise ValueError("Result identity does not match its recorded hash")
        if row.get("qualifying"):
            duration = row.get("performance", {}).get("total_e2e_s")
            if (
                not isinstance(duration, (int, float))
                or isinstance(duration, bool)
                or not math.isfinite(duration)
                or duration <= 0
            ):
                raise ValueError(
                    "A qualifying result must have a positive finite duration"
                )
    return rows


def _qualifying(rows: list[dict], *, old: bool = False) -> list[dict]:
    return [
        r
        for r in rows
        if r.get("qualifying")
        and (not old or r.get("correctness", {}).get("finished_preempted_count") == 0)
    ]


def compare(
    target: Identity,
    reference: Identity,
    results: list[dict],
    *,
    previously_triggered: set[str] | None = None,
    reference_equivalence: dict[str, str] | None = None,
    native_improvement: float = 0.10,
) -> dict:
    if not 0 < native_improvement < 1:
        raise ValueError("Native improvement must be between zero and one")
    if target.case.mode != "offload-agent" or reference.case.mode not in BOUNDARIES:
        raise ValueError("Only the three specified offload-agent comparisons are gates")
    if target.case.qps != reference.case.qps:
        raise ValueError("Cannot compare different QPS points")
    if target.workload_sha256 != reference.workload_sha256:
        raise ValueError("Comparison members have different frozen inputs")
    boundary = (
        1 - native_improvement
        if reference.case.mode == "native"
        else BOUNDARIES[reference.case.mode]
    )
    gate_id = content_hash(
        [target.key, reference.key]
        if boundary == BOUNDARIES[reference.case.mode]
        else [target.key, reference.key, boundary]
    )
    left = measurements(target, results)
    right = measurements(reference, results)
    output: dict[str, Any] = {
        "gate_id": gate_id,
        "qps": target.case.qps,
        "phase": target.case.phase,
        "reference_mode": reference.case.mode,
        "target_identity": target.key,
        "reference_identity": reference.key,
        "maximum_ratio": boundary,
        "status": "unresolved",
        "reason": "missing_qualifying_result",
        "repeat_triggered": gate_id in (previously_triggered or set()),
        "launches": {target.key: len(left), reference.key: len(right)},
        "requested_launches": {},
    }
    if target.case.phase == "phase-2" and right:
        proof = (reference_equivalence or {}).get(reference.key)
        if not proof:
            output["reason"] = "reference_equivalence_required"
            return output
        output["reference_equivalence"] = proof
    old = reference.case.mode == "old-offload-agent"
    valid_left, valid_right = _qualifying(left), _qualifying(right, old=old)
    output["gate_inputs"] = {
        target.key: [r["result_path"] for r in valid_left],
        reference.key: [r["result_path"] for r in valid_right],
    }
    if (
        old
        and not valid_right
        and any(
            r.get("correctness", {}).get("finished_preempted_count", 0) > 0
            for r in right
        )
    ):
        output["status"] = "work_inequivalent_reference"
        output["reason"] = "old_finished_preempted"
        return output
    if not valid_left or not valid_right:
        for identity, rows, valid in (
            (target, left, valid_left),
            (reference, right, valid_right),
        ):
            if not valid:
                if len(rows) < MAX_LAUNCHES:
                    output["requested_launches"][identity.key] = 1
                else:
                    output["reason"] = "exhausted_nonqualifying_budget"
        # An exhausted member cannot be repaired by more work on its peer.
        if output["reason"] == "exhausted_nonqualifying_budget":
            output["requested_launches"].clear()
        return output
    left_time = median(r["performance"]["total_e2e_s"] for r in valid_left)
    right_time = median(r["performance"]["total_e2e_s"] for r in valid_right)
    ratio = left_time / right_time
    initial_ratio = (
        valid_left[0]["performance"]["total_e2e_s"]
        / valid_right[0]["performance"]["total_e2e_s"]
    )
    boundary = output["maximum_ratio"]
    triggered = (
        output["repeat_triggered"]
        or initial_ratio >= boundary - GRAY_BAND
        or ratio >= boundary - GRAY_BAND
    )
    output.update(
        repeat_triggered=triggered,
        target_median_s=left_time,
        reference_median_s=right_time,
        normalized_ratio=ratio,
        initial_ratio=initial_ratio,
    )
    if triggered:
        for identity, rows in ((target, left), (reference, right)):
            if len(rows) < MAX_LAUNCHES:
                output["requested_launches"][identity.key] = MAX_LAUNCHES - len(rows)
    if output["requested_launches"]:
        output["reason"] = "affected_pair_repetition"
    else:
        output["status"] = "pass" if ratio <= boundary else "fail"
        output["reason"] = "strict_median_gate" if triggered else "strict_initial_gate"
    return output


def evaluate(
    identities: list[Identity],
    results: list[dict],
    *,
    include_old: bool = True,
    previously_triggered: set[str] | None = None,
    reference_equivalence: dict[str, str] | None = None,
    native_improvement: float = 0.10,
    qps_values: Sequence[float] = QPS,
) -> dict:
    lookup = {
        (identity.case.mode, identity.case.qps): identity for identity in identities
    }
    if len(lookup) != len(identities):
        raise ValueError(
            "A report must select one implementation identity per mode/QPS"
        )
    gates = []
    unprepared = []
    requested: dict[str, int] = {}
    for qps in qps_values:
        target = lookup.get(("offload-agent", qps))
        for mode in BOUNDARIES:
            if mode == "old-offload-agent" and not include_old:
                continue
            reference = lookup.get((mode, qps))
            if target is None or reference is None:
                unprepared.append(
                    {
                        "qps": qps,
                        "reference_mode": mode,
                        "missing_modes": [
                            name
                            for name in ("offload-agent", mode)
                            if (name, qps) not in lookup
                        ],
                    }
                )
                continue
            gate = compare(
                target,
                reference,
                results,
                previously_triggered=previously_triggered,
                reference_equivalence=reference_equivalence,
                native_improvement=native_improvement,
            )
            gates.append(gate)
            for key, count in gate["requested_launches"].items():
                requested[key] = max(requested.get(key, 0), count)
    references = []
    for qps in qps_values:
        identity = lookup.get(("mooncake", qps))
        if identity is None:
            continue
        rows = measurements(identity, results)
        references.append(
            {
                "qps": qps,
                "gating": False,
                "launches": len(rows),
                "status": "not_run"
                if not rows
                else "qualifying"
                if rows[0].get("qualifying")
                else "excluded",
                "total_e2e_s": rows[0]["performance"]["total_e2e_s"]
                if rows and rows[0].get("qualifying")
                else None,
                "result_path": rows[0]["result_path"] if rows else None,
            }
        )
    return {
        "gates": gates,
        "triggered_gates": sorted(g["gate_id"] for g in gates if g["repeat_triggered"]),
        "requested_launches": requested,
        "mooncake_speed_reference": references,
        "unprepared_comparisons": unprepared,
        "passed": not unprepared and all(g["status"] == "pass" for g in gates),
        "applicable_gates_passed": not unprepared
        and all(g["status"] in ("pass", "work_inequivalent_reference") for g in gates),
        "unresolved_decisions": [
            {"gate_id": g["gate_id"], "reason": g["reason"]}
            for g in gates
            if g["status"] == "work_inequivalent_reference"
            or g["status"] == "unresolved"
            and not g["requested_launches"]
        ]
        + [
            {"reason": "unprepared_comparison", **comparison}
            for comparison in unprepared
        ],
    }
