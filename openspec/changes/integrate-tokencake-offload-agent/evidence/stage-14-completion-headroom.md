# Completion-headroom experiment, 2026-09-06

This candidate commits all admitted prefill growth plus enough generation
capacity for one admitted request to complete. Other requests retain KV while
waiting at block-growth boundaries. Full-attention groups use native multi-group
demand calculation; other cache types and speculative lookahead retain full
generation commitments. A separate gauge records critical growth waits.

Campaign: `/root/autodl-tmp/tokencake-optimization/combined-08`.
Immutable runtime tree:
`aeb7e6936ad2637b8637e6f756bacaa34459242520b38fe9ed1d1b5d3a7c22f0`.
Artifact hash:
`2d9e558228b2e11b570e62e562639e1f58814db846329dd76bbbbda2afa570ad`.
Results: `cases/phase-1/{agent,offload-agent}/qps-1.0/launch-0/result.json`.

| Metric | Agent | Offload-agent |
| --- | ---: | ---: |
| Total E2E seconds | 1456.597680 | 1528.298596 |
| Average application latency seconds | 1286.580258 | 1389.468668 |
| Application p95 seconds | 1423.610815 | 1507.468999 |
| Application p99 seconds | 1438.922146 | 1512.562443 |
| Physical / reservation preemptions | 0 / 0 | 0 / 0 |
| Reservation / capacity denials | 1763 / 1434912 | 2202 / 1478252 |
| Running growth deferrals | 71820 | 98074 |
| Executed / recomputed tokens | 3515292 / 0 | 3513590 / 0 |
| GPU / CPU admission hit tokens | 2145504 / 0 | 2145456 / 1616 |
| Resume GPU / CPU hit tokens | 0 / 0 | 0 / 0 |
| Maximum critical admission wait seconds | 295.502 | 529.862 |
| Maximum critical running growth wait seconds | 8.209 | 6.522 |
| Critical admissions waiting at least 180 seconds | 26 | 34 |
| Steps reporting at most one token / all steps | 6081 / 38647 | 10388 / 42132 |
| Completed saved GPU blocks | 0 | 596 |

Both launches qualify for correctness and isolation. Each completes 24 DAGs,
696 nodes, 648 model calls, and 271200 output tokens. Retries, prompt halvings,
terminal preemptions, and transfer failures are zero. Denials and growth
deferrals count repeated scheduling decisions rather than unique requests.

Against the qualified native median of 1362.916869 seconds, agent is 6.87%
slower and offload-agent is 12.13% slower. Compared with combined-07, total E2E
regresses by 5.73% and 10.81%. Small execution steps and running growth waits
show that preserving every active KV allocation can reduce useful concurrency
even when all requests eventually finish without recomputation. The `progress`
mode is rejected as the next default; it remains available for explicit use.
No 25% gain or completed three-QPS acceptance is claimed.

The source, launcher, model, resources, arrivals, and full workload are unchanged.
Observed workload hash:
`1590183398f961e31fa1253252e7624a5ef8a64d1406c00610a4587f999f60ed`.

Verification before freezing:

- Scheduling, native scheduler, offloading, and configuration: 343 passed in
  79.43 seconds; `.venv/optimization-progress-capacity-final-unit.xml`.
- Real-model FCFS and priority serving: 2 passed in 84.02 seconds;
  `.venv/optimization-progress-capacity-model.xml`.
- Changed-file pre-commit hooks, including type checks: passed.

Commands use `.venv/bin/python -m pytest` with `HF_HUB_OFFLINE=1`. Unit files:
`tests/tokencake/test_scheduling.py`, `tests/tokencake/test_offloading.py`,
`tests/tokencake/test_config.py`, and `tests/v1/core/test_scheduler.py`.
Serving: `tests/tokencake/test_scheduling_serving.py`, with
`CUDA_VISIBLE_DEVICES=1` and
`VLLM_TEST_MODEL=/root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct`.
