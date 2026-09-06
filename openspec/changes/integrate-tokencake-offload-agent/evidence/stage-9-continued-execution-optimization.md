# Continued-execution optimization, 2026-09-06

The user confirmed native vLLM as the reference, a total-E2E reduction of at
least 25% at each of QPS 1.0, 0.5, and 0.1, and combined optimization before
single-factor ablations. Historical campaigns retain their original gates.

## Candidate and provenance

Campaign: `/root/autodl-tmp/tokencake-optimization/combined-02`.

The first candidate moves reservation enforcement to new admission, accounts
for outstanding admitted prefill growth through native multi-group demand,
uses importance-bounded recomputation cost for physical victim selection,
checks capacity improvement before resumption, and restores native prefill
chunking. It allocates KV incrementally through the native block pool.

The immutable runtime snapshot has tree hash
`74ad71398997f05de38290dc17ff1b45f4596b1f6814824915fb35cff62215ee`.
Native uses the unchanged baseline checkout. The dataset, arrival trace, model,
reported 48,800-token KV configuration, NUMA affinities, and full 24-DAG workload are
frozen in `frozen.json` and `provenance.json`. The observed workload hash is
`1590183398f961e31fa1253252e7624a5ef8a64d1406c00610a4587f999f60ed`.

Later working-tree changes do not change this snapshot. In particular, the
subsequent per-step candidate-order cache is not included in these timings.

## Full QPS 1.0 results

| Measurement | Native | Agent scheduling | Offload-agent |
| --- | ---: | ---: | ---: |
| Total E2E, seconds | 1373.340746 | 1446.081098 | 1453.745923 |
| Application p95, seconds | 1337.799399 | 1437.862393 | 1429.277033 |
| Application p99, seconds | 1343.043196 | 1439.791247 | 1442.070112 |
| Completed applications | 24 | 24 | 24 |
| Model calls | 648 | 648 | 648 |
| Generated output tokens | 271200 | 271200 | 271200 |
| Retries / prompt halvings | 0 / 0 | 0 / 0 | 0 / 0 |
| Terminal preemptions | 0 | 0 | 0 |
| Physical preemptions | 54 | 54 | 24 |

All three launches qualify for correctness and environment isolation. Their
individual results are under
`cases/phase-1/{native,agent,offload-agent}/qps-1.0/launch-0/result.json`.
Native and offload-agent use GPU0; agent uses GPU1. The peer state is retained
in each monitor artifact. The scheduling candidate is 5.30% slower than native
and offload-agent is 5.85% slower; the required 25%
reduction would require at most 1030.005560 seconds against this launch.
This is an initial diagnostic comparison, not a completed three-QPS gate.

Agent scheduling additionally records:

- Reservation-triggered preemptions: 0.
- Reservation denials: 757851; aggregate prefill-capacity denials: 656438;
  unchanged-capacity resumption deferrals: 315. These count decisions, not
  distinct requests or preemptions.
- Actual model-returned execution: 3710498 tokens, including 197129 repeated
  positions. Rolled-back scheduling decisions do not count as executed work.
- GPU matched tokens at successful allocation: 2277680, including 123136
  during resumption. CPU matches are zero in the scheduling-only mode.
- Maximum critical admission wait: 683.748 seconds; 301 critical admissions
  waited at least 60 seconds and 44 at least 180 seconds. These include
  readmissions. All requests eventually completed, but long waits remain.

Offload-agent records 24 physical and zero reservation-triggered preemptions,
3597006 executed tokens, 84681 repeated positions, and 62544 GPU tokens matched
during resumption. Reservation denials are 944414, prefill-capacity denials
551762, and unchanged-capacity resumption deferrals 31. Critical admission wait
reaches 683.341 seconds, with 310 observations at least 60 seconds and 37 at
least 180 seconds. All requests eventually complete.

Offload selection, completed preservation blocks, and CPU matches are all
zero. There are 4333 no-waiting-demand decisions and 3272 stale-snapshot/prefix
gap observations. Snapshot-external frontier observations total 853330; these
are repeated observations, not distinct blocks or tokens. No timing change can
be attributed to successful CPU KV reuse in this launch.

QPS 0.5 and 0.1 have not been run for this candidate. The generated
`reports/0000.json` records `passed=false`, the unmet initial native threshold,
the prescribed two additional launches per affected pair member for a formal
median gate, and unprepared comparisons. These repeats have not been launched
while the next policy revision awaits the user's decision. No 25% success
claim is supported by this record.

## Verification

- Focused scheduling/report/driver tests: 83 passed, recorded in
  `.venv/optimization-focused.xml`.
- Real-model FCFS and priority completion tests: 2 passed in 84.561 seconds,
  recorded in `.venv/optimization-scheduling-model.xml`. Each contended
  request completed its full 128-token output.
- Core scheduling/offloading suite: 188 passed and 2 failed in the original
  `.venv/optimization-core.xml`. The failures were a two-GPU test launched
  with only one visible GPU and a manually constructed scheduler fixture
  missing the optional TokenCake state. Both passed on targeted recheck after
  correcting the environment and fixture; the original XML is preserved.
- Changed-file pre-commit checks, including type checks, passed for the frozen
  candidate. Strict OpenSpec validation and `git diff --check` also passed.
- After the timed campaign, the final scheduling, native scheduler, offloading,
  experiment driver/report, and lifecycle/event suite passed all 248 tests in
  58.93 seconds. It includes the new sync/async recovery-hit and frozen-order
  regressions. Artifact: `.venv/optimization-final-unit.xml`.
- The final working-tree real-model FCFS/priority tests passed again: 2 tests
  in 83.57 seconds. Artifact: `.venv/optimization-final-model.xml`. The new
  candidate-order cache is correctness-tested but has no full-DAG timing yet.

Final test commands:

```bash
env HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
  tests/tokencake/test_scheduling.py tests/v1/core/test_scheduler.py \
  tests/tokencake/test_offloading.py tests/tokencake/test_experiment_driver.py \
  tests/tokencake/test_experiment_report.py tests/tokencake/test_events.py \
  -q --junitxml=.venv/optimization-final-unit.xml
env PATH="/root/TokenCake/vLLM-TokenCake-upstream/.venv/bin:$PATH" \
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 \
  VLLM_TEST_MODEL=/root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct \
  .venv/bin/python -m pytest tests/tokencake/test_scheduling_serving.py \
  -q --junitxml=.venv/optimization-final-model.xml
```

## Pending decisions

Two extensions were presented to the user and remain unimplemented pending
their response: logically commit the declared generation budget in addition
to known input, and correct offload waiting-demand estimation so a native
chunk larger than the preservation window is not incorrectly discarded.
Neither is part of this candidate's runtime snapshot.

The subsequent 2026-09-06 user goal authorizes autonomous strategy decisions,
continued iteration to the performance standard, end-to-end tests, and a code
commit after each stage. The two extensions are therefore authorized for a new
candidate; the measurements above remain tied to the unchanged stage-9 snapshot.

## Generation Commitment And Window Correction

The combined-03 snapshot adds complete generation commitments, the cached
per-step candidate order, and the waiting-window correction. Its runtime hash is
`d40d126118eb0c82dcbacc63c352d599de18971fce335ba67cd7c532671c4144`.
The complete QPS 1.0 campaign is
`/root/autodl-tmp/tokencake-optimization/combined-03`; result paths are
`cases/phase-1/{agent,offload-agent}/qps-1.0/launch-0/result.json`.

| Metric | Agent | Offload-agent |
| --- | --- | --- |
| Total E2E seconds | 1455.396173 | 1479.603045 |
| Application p95 seconds | 1425.614287 | 1448.999788 |
| Application p99 seconds | 1438.469621 | 1457.665743 |
| Physical / reservation preemptions | 0 / 0 | 0 / 0 |
| Reservation admission denials | 764632 | 783358 |
| Generation capacity denials | 785679 | 772571 |
| Executed / recomputed tokens | 3507586 / 0 | 3508179 / 0 |
| GPU / CPU admission hit tokens | 2153152 / 0 | 2152560 / 0 |
| Resume GPU / CPU hit tokens | 0 / 0 | 0 / 0 |
| Maximum critical admission wait seconds | 687.224 | 683.581 |
| Critical admissions waiting at least 180 seconds | 45 | 39 |
| Completed saved GPU blocks | 0 | 1272 |

Both results qualify for correctness and isolation: 24 DAGs, 696 nodes, 648
completed model calls, 271200 output tokens, no retry, prompt halving, or
terminal preemption. No new native reference was launched. The previously
qualified native launch-1 and launch-2 results under
`/root/autodl-tmp/tokencake-acceptance/phase-1-repaired` have identical identity
`761ca710418a3c7d7aacbfee4e614219b8b21272bd5132ee6baacd0c4a801a6e`
and median 1362.916869 seconds. Agent is 6.79% slower and offload-agent 8.56%
slower. The 25% reduction target is not met.

Before freezing this candidate, scheduling and native scheduler validation
passed 151 tests (`.venv/optimization-growth-core.xml`), and real-model FCFS and
priority serving passed two tests (`.venv/optimization-growth-model.xml`).
The full changed-file pre-commit checks passed.

Inspection of the native output record found that all 120 validation-to-repair
continuations prepend the role prompt again. They share only 77 or 87 leading
characters with their predecessor, including all 72 offload-eligible pairs.
The user explicitly instructed on 2026-09-06 to keep the workload unchanged.
No workload, prompt, arrival, token-limit, or source-checkout change was made.
The next scheduling candidate addresses idle reservation capacity; this result
does not claim long-prefix offload reuse from metadata alone.
