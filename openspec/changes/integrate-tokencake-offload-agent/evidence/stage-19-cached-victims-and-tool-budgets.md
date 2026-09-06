# CPU-aware victim cost and tool instruction budgets, 2026-09-06

Within the existing importance band, victim selection now subtracts the
currently CPU-restorable prefix from estimated recomputation. It retains
the sufficient-release and near-completion rules. The lookup reuses native
full-attention and hybrid alignment on temporary request state, without
touching CPU cache recency, refcounts or store-frequency tracking. Pending
stores and loads receive no recovery credit. Actual readmission retains
native restoration and recomputation if the CPU cache changes meanwhile.

The new `conversation-tools` profile retains conversation prefixes and limits
13 tool-instruction calls per DAG to 128 output tokens. Planning and review
remain at 200; code validation and repair remain at 400. All 24 DAGs, 696
nodes, 648 model calls, original tool durations and arrivals remain present.
Total declared generation changes from 271200 to 155136 tokens. Native and
TokenCake receive the same new budgets and construction rules. The source
checkout remains clean. No speculative decoding is used.

Campaign: `/root/autodl-tmp/tokencake-optimization/conversation-tools-01`.
Runtime tree:
`26d913fa251bf33ef600e0e1a41d87df7ec1a94d5ef4653bfbef82203b816323`.
Target artifact:
`ada63176125182ac2e581634eac93f26a3d77c6cbfc41937bb2d6644c0c0e946`.
Observed workload:
`e2de7eb7d6187bc0653c3d8967f769d38aa78da7d47043f0e271711ab9b1e65c`.
Frozen input identity:
`3b8f1be50843e2165cd31a0a7c11b27d0a437df1789eca41129307d713f5e171`.
Native case key:
`e0fca89be131e220c71f9b3249c33239594ef58b205cf036c16f4d594edd98b3`.
Offload-agent case key:
`9b3ed24cedd6f7ea9a3f3fd6d39c2c78a901f88e281eb9d10004ff9edbd6b89e`.

Native uses GPU 0; offload-agent uses GPU 1. Each is a fresh process on one
A800 80 GB with declared NUMA affinity and 0.50 GPU memory utilization.
The model, cache geometry and batch limits remain unchanged. Both launches
qualify. Results are `cases/phase-1/native/qps-1.0/launch-0/result.json` and
`cases/phase-1/offload-agent/qps-1.0/gpu-1/launch-0/result.json`.

| Metric | Native | Offload-agent |
| --- | ---: | ---: |
| Total E2E seconds | 945.470247 | 656.552400 |
| Average application latency seconds | 857.192797 | 542.105777 |
| Application p95 seconds | 915.877937 | 627.159962 |
| Application p99 seconds | 917.198406 | 630.390066 |
| Physical / reservation preemptions | 50 / N/A | 66 / 0 |
| Reservation / capacity denials | N/A | 2494 / 460771 |
| Executed / recomputed tokens | N/A | 1347224 / 49781 |
| GPU / CPU admission hit tokens | N/A | 3563312 / 1747840 |
| Resume GPU / CPU hit tokens | N/A | 317920 / 258016 |
| Maximum critical admission wait seconds | N/A | 118.213 |
| Critical admissions waiting at least 180 seconds | N/A | 0 |
| Completed saved GPU blocks | N/A | 18101 |
| Scheduler iteration observations | 17730 | 16267 |

Total E2E improves by 30.56% relative to native on this new workload. The
ratio is below 0.72, passing the configured initial 25% gate without its
three-percentage-point repeat trigger. This validates QPS 1.0 only. It does
not establish the result for QPS 0.5 or 0.1, or for the previous workloads.
The integrated change does not isolate the contribution of victim cost.

Both cases complete all declared work, including 155136 output tokens.
Retries, prompt halvings, terminal preemptions, running growth deferrals and
transfer failures are zero. Native lacks TokenCake's execution-range and
critical-wait instrumentation. Downstream generated context may vary under
the same construction rules; actual prompt totals are 5875700 and 5875419.

Verification before freezing:

- Three CPU-backed victim selection reproductions failed before the change.
  `.venv/optimization-cached-victim-red.xml`.
- 43 focused cache-readiness, read-only lookup, native group and complete
  preemption/restoration tests passed in 11.18 seconds.
  `.venv/optimization-cached-victim-focused.xml`.
- 374 CPU scheduling, offloading, configuration and native offloader tests
  passed in 80.71 seconds; 3 GPU configuration cases were deselected.
  `.venv/optimization-cached-victim-cpu-unit.xml`.
- 4 actual-model scheduling and CPU restoration checks passed in 174.51
  seconds; `.venv/optimization-cached-victim-model.xml`.
- 49 experiment driver, report, launcher and workload checks passed in
  16.49 seconds; `.venv/optimization-tools-budget-driver.xml`.
- Changed-file pre-commit hooks, including type checks, passed.

Tests use `.venv/bin/python -m pytest`. CPU tests set `CUDA_VISIBLE_DEVICES=`.
Model tests use GPU 1, `HF_HUB_OFFLINE=1`, and the frozen local Qwen2.5-14B
model. The workload probe checks every node's budget in both request modes,
all 26 continuation edges, unchanged tool windows and stable join ordering.
The later admission-capacity precheck is absent from this immutable runtime
snapshot and is not validated by this E2E result.
