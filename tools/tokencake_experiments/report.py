# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize observed measurements without acceptance gates or launch budgets."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

from tools.tokencake_experiments.campaign import Case


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
    return sorted(
        (
            row
            for row in results
            if row.get("budget_identity", row["identity"].get("key")) == identity.key
            or row["identity"]["case"] == identity.payload()["case"]
        ),
        key=lambda row: row["launch"],
    )


def summarize(identities: list[Identity], results: list[dict]) -> dict:
    cases = []
    for identity in identities:
        rows = measurements(identity, results)
        durations = [
            value
            for row in rows
            if row.get("qualifying")
            and isinstance(
                value := row.get("performance", {}).get("total_e2e_s"), (int, float)
            )
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
        ]
        cases.append(
            {
                "case": identity.case.payload(),
                "launches": len(rows),
                "measured_launches": len(durations),
                "median_e2e_s": median(durations) if durations else None,
                "result_paths": [row.get("result_path") for row in rows],
                "excluded_launches": [
                    {
                        "launch": row["launch"],
                        "reasons": row.get("exclusion_reasons", []),
                    }
                    for row in rows
                    if not row.get("qualifying")
                ],
            }
        )
    baselines = {
        (row["case"]["phase"], row["case"]["qps"]): row["median_e2e_s"]
        for row in cases
        if row["case"]["mode"] == "native"
    }
    comparisons = []
    for row in cases:
        if row["case"]["mode"] == "native":
            continue
        base = baselines.get((row["case"]["phase"], row["case"]["qps"]))
        current = row["median_e2e_s"]
        comparisons.append(
            {
                "case": row["case"],
                "native_median_e2e_s": base,
                "median_e2e_s": current,
                "e2e_reduction_pct": 100 * (1 - current / base)
                if base and current
                else None,
                "speedup": base / current if base and current else None,
            }
        )
    return {"cases": cases, "comparisons": comparisons}
