# Conversation generation reclaim, 2026-09-06

The default generation reservation mode changes from `all` to `reclaim`.
Known input growth remains committed. Physical allocation failure identifies
a beneficiary and reserves its remaining generation while selecting victims
that release enough capacity. The conversation workload and preservation
implementation are identical to stage 17. No speculative decoding is used.

Campaign: `/root/autodl-tmp/tokencake-optimization/conversation-02`.
Runtime tree:
`f4c105902e035be8f16d725bc7b11b2206057b9ac75988128cd5dea546041d52`.
Target artifact:
`90c9e28018e6a0e691329b0b5b3a42d2e6bc707e183ddb515381b8dd625e5237`.
Observed workload:
`ad3062465acbaa4de1ce508f65ac1bc6bf6818c964d9bc5eb0187ac2c2fbb55b`.
Offload-agent case key:
`488f6d8a05f9a1990bad33112cfcc605d81bfc2952e6f1d6ec0bdcdcd310a7a2`.
Result: `cases/phase-1/offload-agent/qps-1.0/gpu-1/launch-0/result.json`.

The reference is the qualifying native launch from `conversation-01`, with
case key `29b1312450a7d191e121db815f3f808dfcfe5f74906844712f2a562f470c2191`.
Its workload, workload construction, native code and native configuration
remain identical. It was not rerun or charged a new launch identity. The
target uses GPU 1, an A800 80 GB, at 0.50 GPU memory utilization with the
declared NUMA affinity. The native reference uses the corresponding GPU 0.

| Metric | Native reference | Offload-agent |
| --- | ---: | ---: |
| Total E2E seconds | 1504.246461 | 1306.039819 |
| Average application latency seconds | 1361.330853 | 1167.280429 |
| Application p95 seconds | 1472.537853 | 1290.187806 |
| Application p99 seconds | 1475.868126 | 1293.858740 |
| Physical / reservation preemptions | 77 / N/A | 99 / 0 |
| Reservation / capacity denials | N/A | 3613 / 1377579 |
| Executed / recomputed tokens | N/A | 1818592 / 163133 |
| GPU / CPU admission hit tokens | N/A | 4793952 / 1997568 |
| Resume GPU / CPU hit tokens | N/A | 542176 / 419120 |
| Maximum critical admission wait seconds | N/A | 417.990 |
| Critical admissions waiting at least 180 seconds | N/A | 38 |
| Completed saved GPU blocks | N/A | 22546 |
| Scheduler iteration observations | 36099 | 37447 |

The native-relative E2E reduction is 13.18%, below the required 25%.
Compared with stage 17's conservative generation commitments, E2E drops
from 1319.488144 to 1306.039819 seconds, while average application latency,
both tails and maximum critical admission wait increase. This single launch
does not establish a statistically significant improvement over stage 17.
It demonstrates that preempted requests can recover substantial CPU KV and
finish, but their remaining recomputation and scheduling delay still matter.

All 24 DAGs, 696 nodes, 648 model calls and 271200 declared output tokens
complete. Retries, prompt halvings, terminal preemptions, running growth
deferrals and transfer failures are zero. Denials count repeated decisions.
Native lacks TokenCake's detailed execution-range and critical-wait metrics.

Verification before freezing:

- 272 CPU scheduling, offloading and configuration tests passed in 61.67
  seconds; 3 GPU configuration checks were deselected.
  `.venv/optimization-conversation-reclaim-cpu-unit.xml`.
- 4 actual-model scheduling and CPU restoration checks passed in 180.47
  seconds; `.venv/optimization-conversation-reclaim-model.xml`.
- Changed-file pre-commit hooks passed.

Tests use `.venv/bin/python -m pytest`. CPU tests set `CUDA_VISIBLE_DEVICES=`.
Model tests use GPU 1, `HF_HUB_OFFLINE=1`, and the frozen local Qwen2.5-14B
model. The later CPU-aware victim implementation was absent from this
campaign's immutable runtime snapshot and is not validated by this result.
