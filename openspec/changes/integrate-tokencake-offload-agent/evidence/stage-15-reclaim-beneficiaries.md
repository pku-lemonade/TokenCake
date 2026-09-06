# Reclaim-beneficiary experiment, 2026-09-06

This candidate commits admitted input growth and gives requests encountering
physical allocation failure a generation commitment. New admissions respect
that commitment; already running requests retain native allocation progress.
Victim selection considers the beneficiary's outstanding commitment when
preferring a candidate that releases enough space. The request continues through
native preemption and resumption rather than receiving a terminal response.

Campaign: `/root/autodl-tmp/tokencake-optimization/combined-09`.
Immutable runtime tree:
`dadfb025419db4addc4692f2c5d6bafa9aa9329441af25d90e1efcf50ed700c0`.
Artifact hash:
`d716a87794112201fec7c3346b83fed7549fb5d8a42c36b61d74bab2f92ce80a`.
Results: `cases/phase-1/{agent,offload-agent}/qps-1.0/launch-0/result.json`.

| Metric | Agent | Offload-agent |
| --- | ---: | ---: |
| Total E2E seconds | 1389.442909 | 1383.236790 |
| Average application latency seconds | 1222.857962 | 1241.134857 |
| Application p95 seconds | 1361.849317 | 1367.925381 |
| Application p99 seconds | 1370.017446 | 1368.479028 |
| Physical / reservation preemptions | 66 / 0 | 66 / 0 |
| Reclaim beneficiaries | 57 | 57 |
| Reservation / capacity denials | 3399 / 1162346 | 1938 / 1271739 |
| Running growth deferrals | 0 | 0 |
| Executed / recomputed tokens | 3835926 / 303804 | 3808103 / 277502 |
| GPU / CPU admission hit tokens | 2324848 / 0 | 2350464 / 2320 |
| Resume GPU / CPU hit tokens | 188080 / 0 | 212800 / 0 |
| Maximum critical admission wait seconds | 413.951 | 374.158 |
| Critical admissions waiting at least 180 seconds | 23 | 41 |
| Completed saved GPU blocks | 0 | 449 |

Both launches qualify and complete all 24 DAGs, 696 nodes, 648 model calls,
and 271200 output tokens. Terminal preemptions are zero. Denial and admission
counters include repeated decisions and resumed requests.

Against the qualified native median of 1362.916869 seconds, agent is 1.95%
slower and offload-agent is 1.49% slower. Eliminating running growth pauses
recovers most of combined-08's regression but introduces substantial repeated
computation. Combined-07 remains faster and has lower critical admission waits.
The next candidate restores the conservative `all` generation commitment as
the default. This experiment does not establish a 25% improvement.

This campaign uses the unchanged original workload. Observed workload hash:
`1590183398f961e31fa1253252e7624a5ef8a64d1406c00610a4587f999f60ed`.
After this campaign started, the user authorized workload changes and explicitly
excluded speculative decoding. Any revised workload requires its own native
baseline and cannot use these measurements to claim a relative speedup.

Verification before freezing:

- Scheduling, offloading, and configuration CPU checks: 263 passed, 3 GPU
  configuration checks deselected, in 60.81 seconds;
  `.venv/optimization-reclaim-cpu-unit.xml`.
- Native scheduler and GPU configuration checks: 100 passed in 25.10 seconds;
  `.venv/optimization-reclaim-native-unit.xml`.
- Real-model FCFS and priority serving: 2 passed in 84.77 seconds;
  `.venv/optimization-reclaim-model.xml`.
- Changed-file pre-commit hooks, including type checks: passed.

Commands use `.venv/bin/python -m pytest` and `HF_HUB_OFFLINE=1`. Unit files:
`tests/tokencake/test_scheduling.py`, `tests/tokencake/test_offloading.py`,
`tests/tokencake/test_config.py`, and `tests/v1/core/test_scheduler.py`.
Serving: `tests/tokencake/test_scheduling_serving.py`, with
`CUDA_VISIBLE_DEVICES=1` and
`VLLM_TEST_MODEL=/root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct`.
