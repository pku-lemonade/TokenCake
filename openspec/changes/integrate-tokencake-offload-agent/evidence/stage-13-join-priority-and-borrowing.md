# Join priority and early borrowing, 2026-09-06

This candidate propagates the highest live agent score within an application's
join group. A request with a 500-point score lead may borrow idle reservations
before a lower-priority owner consumes capacity. Smaller leads retain owner
priority. Shared-prefix admission preference now requires shared physical blocks
to cover at least the new capacity demand, excluding short common headers.
The complete generation budget remains committed for every admitted request.

Campaign: `/root/autodl-tmp/tokencake-optimization/combined-07`.
Immutable runtime tree:
`da03e91c210b6ada8d225b8ce19730671058f18a73251a68b66d7d0aa7c0004c`.
Artifact hash:
`ceb137bd006afc4bcee1cceccff834f522bc652fbe29e0e4c1e4d7d519fdf761`.
Results: `cases/phase-1/{agent,offload-agent}/qps-1.0/launch-0/result.json`.

| Metric | Agent | Offload-agent |
| --- | ---: | ---: |
| Total E2E seconds | 1377.694103 | 1379.144211 |
| Average application latency seconds | 1152.138470 | 1164.326053 |
| Application p95 seconds | 1348.497440 | 1352.844704 |
| Application p99 seconds | 1350.972341 | 1353.683901 |
| Physical / reservation preemptions | 0 / 0 | 0 / 0 |
| Reservation / generation-capacity denials | 3174 / 1183545 | 3595 / 1153943 |
| Idle / priority borrowing admissions | 169 / 57 | 184 / 55 |
| Shared-prefix / join-priority admissions | 79 / 304 | 84 / 306 |
| Executed / recomputed tokens | 3529439 / 0 | 3526109 / 0 |
| GPU / CPU admission hit tokens | 2131344 / 0 | 2130144 / 4416 |
| Shared / exclusive GPU hit blocks | 132470 / 739 | 132010 / 1124 |
| Resume GPU / CPU hit tokens | 0 / 0 | 0 / 0 |
| Maximum critical admission wait seconds | 220.508 | 219.542 |
| Critical admissions waiting at least 180 seconds | 8 | 7 |
| Completed saved GPU blocks | 0 | 591 |

Both launches qualify for correctness and isolation. Each completes 24 DAGs,
696 nodes and 648 model calls. Retries, prompt halvings, terminal preemptions,
and transfer failures are zero. Admission denials count repeated scheduling
decisions rather than unique requests.

Against the qualified native median of 1362.916869 seconds, agent is 1.08%
slower and offload-agent is 1.19% slower. Total E2E has not improved. Relative
to combined-06, maximum critical admission waits fall from 631.639/617.702 to
220.508/219.542 seconds, and admissions waiting at least 180 seconds fall from
51/45 to 8/7. These fairness improvements do not satisfy the 25% E2E target.
Three-QPS acceptance has not been completed.

The source, launcher, model, resources, arrivals, and full workload are unchanged.
Observed workload hash:
`1590183398f961e31fa1253252e7624a5ef8a64d1406c00610a4587f999f60ed`.
A nonblocking external CPU sampling attempt failed before attachment because
the container denied process memory access. A separate CPU-only scheduler
diagnostic measured 2.15 ms per step with 10 running and 62 waiting requests;
it did not execute model work and is not an E2E performance claim.

Verification before freezing:

- Scheduling, native scheduler, offloading, and configuration: 316 passed in
  71.70 seconds; `.venv/optimization-join-borrow-unit.xml`.
- Real-model FCFS and priority serving: 2 passed in 84.28 seconds;
  `.venv/optimization-join-borrow-model.xml`.
- Changed-file pre-commit hooks, including type checks: passed.

Commands use `.venv/bin/python -m pytest` with `HF_HUB_OFFLINE=1`. Unit files:
`tests/tokencake/test_scheduling.py`, `tests/tokencake/test_offloading.py`,
`tests/tokencake/test_config.py`, and `tests/v1/core/test_scheduler.py`.
Serving: `tests/tokencake/test_scheduling_serving.py`, with
`CUDA_VISIBLE_DEVICES=1` and
`VLLM_TEST_MODEL=/root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct`.
