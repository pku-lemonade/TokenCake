# Phase-1 Execution: Reservation Progress Repair

The initial full-DAG campaign at
`/root/autodl-tmp/tokencake-acceptance/phase-1-final` exposed a running-request
progress bug that the smaller model pressure tests had not reproduced. Native
baseline and target agent-only QPS 1.0 both loaded the complete 14B model and
started all 24 DAG applications. Agent-only reached 17 running requests,
7 waiting/deferred requests, 99.5% KV usage and sustained zero model throughput.
Neither case had client retries or third-party GPU contamination at diagnosis.

Running reservation denial previously skipped a request without freeing its
blocks. All running borrowers could be denied while a waiting reserved owner
required more physical blocks than were free. The source's running branch sends
this denial through preemption. The migrated branch now does the same using
native `_preempt_request()`, rollback, requeue and recomputation. Waiting
reservation denial continues to defer. No terminal preemption or relaxed
reservation policy was introduced.

Both active launches were stopped through the campaign's SIGTERM handler;
the controlled server/client process groups were cleaned up. Their original
`result.json`, `execution.json`, commands, full configuration, client/server
logs and GPU/process/thermal samples remain under the old campaign directory.
Both are excluded, including the interrupted healthy baseline. No truncated
timing is used for acceptance. The initial driver recorded zero primary time
for interrupted clients; this is non-qualifying evidence only. The driver now
also records the actual interrupted process interval and observed server state.

Recovery preserves the original excluded results and their old implementation
identities, and explicitly charges those launches against the corrected
phase/mode/QPS budget. It cannot transfer qualifying timings or change the
workload/configuration/environment contract. Consequently native and agent-only
QPS 1.0 have each already spent one of their three allowed launches. Phase 2
has not been triggered; this is a Phase-1 correctness repair.

## Verification

The deterministic native-scheduler reproducer covers both FCFS and priority:
a running borrower occupies shared blocks, a waiting owner cannot fit in the
remaining reserved blocks, native preemption frees capacity, the owner runs,
and the borrower subsequently recomputes. It does not mock the allocator.

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
  tests/tokencake/test_experiment_driver.py \
  tests/tokencake/test_experiment_report.py \
  tests/tokencake/test_scheduling.py tests/v1/core/test_scheduler.py \
  -k 'experiment or scheduling or preempt' -q \
  --junitxml=.venv/stage-8-recovery.xml
PATH="$PWD/.venv/bin:$PATH" HF_HUB_OFFLINE=1 \
  VLLM_TEST_MODEL=/root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct \
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/tokencake/test_scheduling_serving.py -q \
  --junitxml=.venv/stage-8-reservation-model.xml
```

The first command passed 96 tests in 25.24 seconds. The second passed both
real-model FCFS/priority contention, preemption and output-recovery cases in
86.52 seconds. XML and output are copied alongside this report. Changed-file
pre-commit checks passed. Full 24-DAG acceptance must still be completed with
the corrected runtime. AI assistance was used.
