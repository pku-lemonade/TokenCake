# Decode-pressure prefill experiment, 2026-09-06

This candidate limits each step to 1024 annotated prefill tokens while decode
requests run. Pure prefill retains the native budget. Encoder input, non-chunked
prefill, and Mamba alignment retain native handling. Preempting an already
scheduled partial prefill refunds both its native and prefill token budgets.

Campaign: `/root/autodl-tmp/tokencake-optimization/combined-05`.
Immutable runtime tree:
`aaf84a09283a982883d317c849f19202dbd26616d1f4a829802802195166b30c`.
Artifact hash:
`6e1b8bbd3611e3799fb17d2b4e5b44c8b2a49d68e684875b70d3cce85c4ed7e2`.
Results: `cases/phase-1/{agent,offload-agent}/qps-1.0/launch-0/result.json`.

| Metric | Agent | Offload-agent |
| --- | ---: | ---: |
| Total E2E seconds | 1471.510323 | 1472.710493 |
| Application p95 seconds | 1446.429826 | 1457.770518 |
| Application p99 seconds | 1450.991480 | 1460.770495 |
| Physical / reservation preemptions | 0 / 0 | 0 / 0 |
| Prefill capping decisions | 1327369 | 1276843 |
| Borrowing admissions | 404 | 400 |
| Reservation / generation-capacity denials | 10095 / 1212368 | 10560 / 1218913 |
| Executed / recomputed tokens | 3522552 / 0 | 3516470 / 0 |
| GPU / CPU admission hit tokens | 2138144 / 0 | 2142576 / 2864 |
| Resume GPU / CPU hit tokens | 0 / 0 | 0 / 0 |
| Maximum critical admission wait seconds | 640.431 | 369.637 |
| Critical admissions waiting at least 180 seconds | 32 | 21 |
| Completed saved GPU blocks | 0 | 436 |

Both launches qualify for correctness and isolation. Each completes 24 DAGs,
696 nodes, 648 model calls, and 271200 output tokens. Retries, prompt halvings,
terminal preemptions, and transfer failures are zero. Capping and admission
denials count repeated scheduling decisions, not unique requests or GPU batches.

Against the existing qualified native median of 1362.916869 seconds, agent is
7.97% slower and offload-agent is 8.06% slower. Compared with combined-04, E2E
regresses by 7.20% and 5.01%. The 1024-token setting is therefore rejected as a
default for the next candidate, which restores the native budget with a zero
setting. The optional pressure limit remains available for later experiments.
No 25% improvement or completed three-QPS acceptance is claimed.

The source, launcher, model, resources, arrivals, and full workload are unchanged.
Observed workload hash:
`1590183398f961e31fa1253252e7624a5ef8a64d1406c00610a4587f999f60ed`.
The draft shared-KV score-band preference is not in this measured snapshot.

Verification before freezing:

- Scheduling, native scheduler, offloading, and configuration: 277 passed and
  two fixture failures in `.venv/optimization-prefill-pressure-unit.xml`.
  The two fixtures incorrectly used 4096-token inputs with a 2048-token model
  limit. After using valid lengths, all nine affected pressure tests passed
  in `.venv/optimization-prefill-pressure-recheck.xml`.
- Real-model FCFS and priority serving: 2 passed in 83.65 seconds, with a
  positive pressure limit exercised and every output complete;
  `.venv/optimization-prefill-pressure-model.xml`.
- Changed-file pre-commit hooks, including type checks: passed.

The test commands use `.venv/bin/python -m pytest` with
`HF_HUB_OFFLINE=1`. The unit selection is `tests/tokencake/test_config.py`,
`tests/tokencake/test_scheduling.py`, `tests/tokencake/test_offloading.py`, and
`tests/v1/core/test_scheduler.py`. The real-model selection is
`tests/tokencake/test_scheduling_serving.py`, using `CUDA_VISIBLE_DEVICES=1` and
`VLLM_TEST_MODEL=/root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct`.
