# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Versioned, identical continuation inputs for native and TokenCake campaigns."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.tokencake_experiments.campaign import ROOT, SOURCE
from tools.tokencake_experiments.materialize import git, materialize


@pytest.mark.parametrize("profile", ["continuation", "conversation"])
def test_revised_profile_is_frozen_and_runs_the_complete_dag(tmp_path, profile):
    if not SOURCE.exists():
        pytest.skip("Requires the frozen source checkout")
    target = materialize(SOURCE, tmp_path / "target", workload_profile=profile)
    reference = materialize(
        SOURCE, tmp_path / "reference", patched=False, workload_profile=profile
    )
    assert target["workload_patch_sha256"] == reference["workload_patch_sha256"]
    assert target["workload_profile"] == reference["workload_profile"] == profile
    name = "agent/app/code_writer_paper_pressure.py"
    assert (
        target["materialized_helpers"][name] == reference["materialized_helpers"][name]
    )
    assert target["materialized_helpers"][name] != target["source_helpers"][name]
    assert git(SOURCE, "status", "--porcelain") == ""
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("experiment_workload_probe.py")),
            target["checkout"],
            profile,
        ],
        cwd=ROOT,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT)},
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "model_calls_per_mode": 27,
        "identical_call_budgets": True,
    }


def test_unknown_workload_profile_cannot_materialize(tmp_path):
    with pytest.raises(ValueError, match="workload profile"):
        materialize(SOURCE, tmp_path / "target", workload_profile="unknown")
