# Stage 6: Launcher And Acceptance Driver

Implemented the reproducible source materializer, minimal protocol patch,
version-compatible client wrapper, fixed two-queue campaign, qualification,
artifact audit, strict median gates and bounded repeat ledger. The runtime vLLM
implementation remains the Stage-5 artifact for this phase.

## Validation

```bash
.venv/bin/python -m pytest \
  tests/tokencake/test_experiment_launcher.py \
  tests/tokencake/test_experiment_driver.py \
  tests/tokencake/test_experiment_report.py -q \
  --junitxml=.venv/stage-6-final.xml
.venv/bin/python -m tools.tokencake_experiments.driver plan \
  --output .venv/stage-6-plan.json
```

The final test run passed 37 tests in 12.71 seconds. Tests execute the actual
materialized launcher against a local HTTP peer, including a complete small
application, four-attempt retry exhaustion, new UUIDs and prompt halving, both
event barriers, HTTP event failure and unchanged tool sleep across modes. This
validates protocol integration and is not model performance acceptance.

The unchanged source analyzer audited a synthetic completed 24-application case
from persisted artifacts, retaining application percentiles and token counts.
Current Mooncake positive-put telemetry is accepted only without connector
errors; raw source analysis remains available. Other checks cover real process
affinity and child cleanup, interrupted launches, contamination, old terminal
preemption, every gate boundary, affected-pair reuse, excluded launch budgets,
Mooncake's once-only rule, and Phase-2 identity/reference-proof requirements.

The planning subprocess produced exactly fifteen initial cases in the accepted
order without launching a model server. An attempted truncation argument was
rejected by its parser. All changed-file pre-commit checks passed, including
Ruff, typos, Python 3.10 mypy, SPDX and forbidden-import checks.

Full workload construction also ran independently through both environments:

```bash
.venv/bin/python tools/tokencake_experiments/source_adapter.py freeze \
  --checkout .venv/launcher-target-v2 --input .venv/workload-parameters.json \
  --output .venv/target-workload.json
.venv/source/.venv/bin/python tools/tokencake_experiments/source_adapter.py freeze \
  --checkout ../vllm_agent --input .venv/workload-parameters.json \
  --output .venv/source-workload.json
cmp .venv/target-workload.json .venv/source-workload.json
```

Both serialized 24-DAG contracts and three arrival traces have SHA-256
`e57a762bca5e0bf97baa0afb6c5ef560e40c1f967798ce4a9a8135b0afd8afd3`.
The protocol patch SHA-256 is
`07613eb608f333267cca69d6f0e624e60079ea11191d9c94193f3d8f10232235`.
Repeated independent materialization produced identical helper hashes and the
original source worktree is clean. The initial fixture failed because it did
not enter the source working directory; that test setup was corrected before
the passing run. No source runtime or DAG implementation was modified.
Removing one empty patch-context line to satisfy `git diff --check` preserved
the materialized code; the seven launcher integration tests were rerun with
`--junitxml=openspec/changes/integrate-tokencake-offload-agent/evidence/stage-6-patch-final.xml`
and passed. The earlier environment setup documentation is retained in the
expanded driver README.

The Mooncake reference pool is explicitly 100 GiB with a 1 GiB local buffer,
retaining the frozen configuration's TCP/P2PHANDSHAKE transport. This sizing
and its rationale are recorded in the driver README and frozen case identity.

Unit and synthetic evidence support this stage only. The full model-based
24-DAG matrix and all performance decisions remain pending. AI assistance was
used; the final diff and acceptance evidence require human review.
