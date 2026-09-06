# Work-conserving borrowing, 2026-09-06

After normal admissions have checked their reservations in agent-score order,
requests denied only by reservation policy receive a second admission pass using
idle reservations. Full native multi-group demand and existing logical growth
commitments still gate admission. Running requests retain their progress.

The complete QPS 1.0 campaign is
`/root/autodl-tmp/tokencake-optimization/combined-04`. Its immutable runtime tree is
`00420040f172713797671305355a5f4469b9c4247973eab2dd23eee413fd7324`.
The artifact hash is
`d2f2dffd277ab8e2d15071432e9fd5fb5df481ad10ebce6bde05ad11a4795491`.
Results are under `cases/phase-1/{agent,offload-agent}/qps-1.0/launch-0/result.json`.

| Metric | Agent | Offload-agent |
| --- | ---: | ---: |
| Total E2E seconds | 1372.692087 | 1402.493786 |
| Application p95 seconds | 1343.562300 | 1365.938674 |
| Application p99 seconds | 1348.932897 | 1374.810901 |
| Borrowing admissions | 347 | 302 |
| Physical / reservation preemptions | 0 / 0 | 0 / 0 |
| Reservation / generation-capacity denials | 8385 / 1264171 | 8440 / 1336695 |
| Executed / recomputed tokens | 3519786 / 0 | 3520806 / 0 |
| GPU / CPU admission hit tokens | 2140800 / 0 | 2136912 / 2992 |
| Resume GPU / CPU hit tokens | 0 / 0 | 0 / 0 |
| Maximum critical admission wait seconds | 218.067 | 596.830 |
| Critical admissions waiting at least 180 seconds | 7 | 36 |
| Completed saved GPU blocks | 0 | 540 |

Both launches qualify for correctness and isolation: 24 complete DAGs, 696 nodes,
648 model calls, and 271200 output tokens. Retries, prompt halvings, terminal
preemptions, and transfer failures are zero. Every accepted request completes.
Admission-denial counters count repeated decisions, not distinct requests.

Against the previously qualified native median of 1362.916869 seconds, agent is
0.72% slower and offload-agent is 2.90% slower. Compared with combined-03, their
E2E improves by 5.68% and 5.21%. The required 25% reduction is not achieved.
QPS 0.5 and 0.1 have not been run for this candidate; no further native launch was
used. Offload has actual CPU hits, but no net E2E benefit in this comparison.

The source workload remains unchanged, as explicitly required by the user.
Its observed hash remains
`1590183398f961e31fa1253252e7624a5ef8a64d1406c00610a4587f999f60ed`.
The draft decode-pressure prefill budget is not part of this measured snapshot.

Verification before freezing:

- Scheduling, native scheduler, and offloading: 208 passed in 52.58 seconds;
  `.venv/optimization-borrowing-unit.xml`.
- Real-model FCFS and priority serving: 2 passed in 87.73 seconds;
  `.venv/optimization-borrowing-model.xml`.
- Changed-file pre-commit hooks, including type checks: passed.

Commands:

```bash
env HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
  tests/tokencake/test_scheduling.py tests/tokencake/test_offloading.py \
  tests/v1/core/test_scheduler.py -q \
  --junitxml=.venv/optimization-borrowing-unit.xml
env PATH="/root/TokenCake/vLLM-TokenCake-upstream/.venv/bin:$PATH" \
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 \
  VLLM_TEST_MODEL=/root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct \
  .venv/bin/python -m pytest tests/tokencake/test_scheduling_serving.py -q \
  --junitxml=.venv/optimization-borrowing-model.xml
```
