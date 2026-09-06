# Shared-prefix admission experiment, 2026-09-06

This candidate restores the native prefill budget by default and prefers
requests sharing active physical KV blocks within 500-point agent score bands
under capacity pressure. Admission still accounts for complete input and
declared generation growth through the native multi-group demand calculation.
GPU hit metrics distinguish active shared blocks from exclusively acquired
cached blocks. Join-priority inheritance is not in this measured snapshot.

Campaign: `/root/autodl-tmp/tokencake-optimization/combined-06`.
Immutable runtime tree:
`ba46157bad43618464786ffc3451f2fc8f1c33157e67a7d3b9e95933cc765236`.
Artifact hash:
`5b13a09ccba5b47115152253e0224fe7fc61298a87becb4a0432386b5c063003`.
Results: `cases/phase-1/{agent,offload-agent}/qps-1.0/launch-0/result.json`.

| Metric | Agent | Offload-agent |
| --- | ---: | ---: |
| Total E2E seconds | 1370.689684 | 1379.141876 |
| Application p95 seconds | 1348.896115 | 1360.790677 |
| Application p99 seconds | 1357.115126 | 1368.777404 |
| Physical / reservation preemptions | 0 / 0 | 0 / 0 |
| Reservation / generation-capacity denials | 8891 / 1359749 | 8973 / 1346144 |
| Idle borrowing admissions | 277 | 323 |
| Shared-prefix preference admissions | 309 | 341 |
| Executed / recomputed tokens | 3514804 / 0 | 3513611 / 0 |
| GPU / CPU admission hit tokens | 2145920 / 0 | 2144368 / 3008 |
| Shared / exclusive GPU hit blocks | 133659 / 461 | 133593 / 430 |
| Resume GPU / CPU hit tokens | 0 / 0 | 0 / 0 |
| Maximum critical admission wait seconds | 631.639 | 617.702 |
| Critical admissions waiting at least 180 seconds | 51 | 45 |
| Completed saved GPU blocks | 0 | 551 |

Both launches qualify for correctness and isolation. Each completes 24 DAGs,
696 nodes, 648 model calls, and 271200 output tokens. Retries, prompt halvings,
terminal preemptions, and transfer failures are zero. Admission denials count
repeated scheduling decisions, not unique requests or GPU batches.

Against the qualified native median of 1362.916869 seconds, agent is 0.57%
slower and offload-agent is 1.19% slower. Compared with combined-04, agent is
0.15% faster and offload-agent is 1.67% faster. These single-run differences
do not establish a reliable gain from shared-prefix ordering. Almost all GPU
hits already share active blocks, and long critical waits remain. The 25%
target and three-QPS acceptance have not been met.

The source, launcher, model, resources, arrivals, and full workload are unchanged.
Observed workload hash:
`1590183398f961e31fa1253252e7624a5ef8a64d1406c00610a4587f999f60ed`.

Verification before freezing:

- Scheduling, native scheduler, offloading, and configuration: 295 passed in
  67.55 seconds; `.venv/optimization-cache-affinity-final-unit.xml`.
- Real-model FCFS and priority serving: 2 passed in 83.40 seconds, with complete
  outputs; `.venv/optimization-cache-affinity-model.xml`.
- Changed-file pre-commit hooks, including type checks: passed.

Commands use `.venv/bin/python -m pytest` with `HF_HUB_OFFLINE=1`. The unit
selection is `tests/tokencake/test_scheduling.py`,
`tests/tokencake/test_offloading.py`, `tests/tokencake/test_config.py`, and
`tests/v1/core/test_scheduler.py`. The serving selection is
`tests/tokencake/test_scheduling_serving.py`, using `CUDA_VISIBLE_DEVICES=1` and
`VLLM_TEST_MODEL=/root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct`.
