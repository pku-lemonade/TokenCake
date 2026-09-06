# Capacity precheck and native acceptance, 2026-09-06

Waiting requests now check logical admission capacity before querying the CPU
cache when all cache groups use full attention, without lookahead or encoder
inputs. The check reuses native multi-group demand and shared GPU prefix hits.
A CPU-restorable prefix requires the same GPU capacity as computing that
prefix. Requests that cannot fit therefore wait without repeatedly searching
or touching CPU cache entries. Once capacity becomes available, native lookup
and asynchronous restoration proceed. Already admitted requests retain their
existing progress and restoration paths; hybrid caches retain native lookup.
This adds no allocation, cache pool, reservation preemption or speculative
decoding. Importance ordering, dynamic reservations and borrowing, and KV
preservation within tool windows remain enabled.

## Workload and provenance

Acceptance uses the authorized `conversation-tools` workload. It retains all
24 DAGs, 696 execution nodes, 648 model calls, original tool durations and
arrival traces. Conversation prefixes are retained across tools and joins.
The 13 tool-instruction calls per DAG use 128 output tokens; planning and
review retain 200, and code validation and repair retain 400. Total declared
generation is 155136 tokens, compared with 271200 in the original workload.
Both native and TokenCake use these same revised construction rules and
budgets. These results do not establish a 25% gain on the original workload,
or equivalent application output quality after reducing tool budgets.

The final target is one immutable runtime for all three QPS points:

- Campaign: `/root/autodl-tmp/tokencake-optimization/conversation-tools-02`.
- Runtime tree:
  `5be699b164a10cadf4e6b8d18847eb42c6ee8aa502305a2a771e98af79aec9d7`.
- Target artifact:
  `b6cd62f97601cd9030bd9f9ade18cd8385eaaf75448a7611ff6c5344a1e2f4de`.
- Observed workload:
  `e2de7eb7d6187bc0653c3d8967f769d38aa78da7d47043f0e271711ab9b1e65c`.
- Frozen input identity:
  `3b8f1be50843e2165cd31a0a7c11b27d0a437df1789eca41129307d713f5e171`.
- Launcher:
  `92615089f08b9e709f394c235a0e885376f5030e9f1e159961e43a9ca00fa242`.
- Native commit: `0b3ba88f165976e77ca5e6a7a3f5bba4562b80af`.

Native QPS 1.0 reuses the qualifying launch from
`/root/autodl-tmp/tokencake-optimization/conversation-tools-01`, whose native
artifact, environment, configuration, launcher and frozen inputs match.
Native QPS 0.5 and 0.1 and all target launches are in `conversation-tools-02`.
Each launch uses a fresh process. Native uses GPU 0 and TokenCake GPU 1,
each one A800 80 GB with declared NUMA affinity, GPU memory utilization 0.50,
Qwen2.5-14B-Instruct, max model length 32768 and native batch limits.
TokenCake retains the existing 100 GiB native CPU offload capacity.

## Formal results

All three native-relative comparisons pass. QPS 1.0 and 0.5 pass their initial
gates; QPS 0.1 passes the three-launch median gate. A target/native E2E ratio
at least 0.72 triggers repeats of the affected pair, with at most three
launches per identity, including exclusions; the final median must be at most
0.75. Ratios below 0.72 pass the initial gate without additional launches.

| QPS | Native E2E seconds | TokenCake E2E seconds | Reduction |
| --- | ---: | ---: | ---: |
| 1.0 | 945.470247 | 660.102569 | 30.18% |
| 0.5 | 955.767485 | 676.604170 | 29.21% |
| 0.1, median | 941.671493 | 688.055419 | 26.93% |

QPS 0.1 target launches are 707.104799, 688.055419 and 682.954863 seconds,
with median 688.055419 seconds. Native launches complete in 939.964527,
941.671493 and 944.819299 seconds, with median 941.671493 seconds. Each
launch counts toward the same original identity budget. The initial 24.77%
reduction is retained in the median calculation. All ten comparison launches
qualify, with no exclusions. No further launches are requested.

The joined [native acceptance report](stage-20-native-acceptance.json) records
all three gates, immutable identity keys, raw result paths and per-launch
observations, with `all_qps_pass=true`. It uses the existing report module's
`compare(..., native_improvement=0.25)` and identity validation. QPS 1.0
selects the native identity from `conversation-tools-01`; the other identities
come from `conversation-tools-02`. This acceptance covers the requested
native comparisons, not the older OpenSpec agent/old-offloader matrix.

## Per-launch observations

Run numbers below start at one. Application percentiles are calculated within
each completed 24-DAG run. Total E2E includes the complete client process;
the client's printed inner-loop duration is not substituted for it.

| QPS | Mode | Run | Average app seconds | App p95 seconds | App p99 seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| 1.0 | Native | 1 | 857.192797 | 915.877937 | 917.198406 |
| 1.0 | TokenCake | 1 | 534.338273 | 632.975702 | 635.125926 |
| 0.5 | Native | 1 | 787.999204 | 903.711775 | 904.122267 |
| 0.5 | TokenCake | 1 | 576.004585 | 633.667788 | 641.240668 |
| 0.1 | Native | 1 | 566.015150 | 723.590277 | 726.376049 |
| 0.1 | Native | 2 | 559.209200 | 730.776729 | 736.388443 |
| 0.1 | Native | 3 | 565.916858 | 730.617348 | 733.858705 |
| 0.1 | TokenCake | 1 | 421.151091 | 502.028922 | 505.802630 |
| 0.1 | TokenCake | 2 | 419.377800 | 511.652273 | 516.132750 |
| 0.1 | TokenCake | 3 | 377.093081 | 458.498162 | 471.061931 |

TokenCake counters below distinguish admission denials from preemptions.
Denials count scheduling decisions, including repeated checks of a waiting
request, rather than distinct requests. Recomputed tokens count overlapping
positions actually executed by the model runner, rather than estimated
victim costs. Native does not expose these TokenCake counters; their absence
is unavailable data, not zero. Native physical preemptions are 50 at QPS 1.0,
38 at QPS 0.5, and 43/32/39 for the three QPS 0.1 runs.

| QPS/run | Capacity / reservation denials | Physical / reservation preemptions | Executed / recomputed tokens | Critical admission wait max seconds |
| --- | ---: | ---: | ---: | ---: |
| 1.0/1 | 419405 / 1761 | 49 / 0 | 1394596 / 48571 | 106.141 |
| 0.5/1 | 498722 / 2424 | 70 / 0 | 1522685 / 60520 | 129.290 |
| 0.1/1 | 297386 / 1526 | 58 / 0 | 1458594 / 77989 | 114.545 |
| 0.1/2 | 310905 / 1642 | 61 / 0 | 1369145 / 78285 | 113.714 |
| 0.1/3 | 243576 / 1341 | 54 / 0 | 1272401 / 57337 | 81.118 |

| QPS/run | GPU admission hit tokens | CPU admission hit tokens | Resume GPU hit tokens | Resume CPU hit tokens |
| --- | ---: | ---: | ---: | ---: |
| 1.0/1 | 3442576 | 1649296 | 221984 | 183664 |
| 0.5/1 | 3766448 | 1407872 | 407744 | 185664 |
| 0.1/1 | 3785600 | 1284896 | 214320 | 196576 |
| 0.1/2 | 3788752 | 1428928 | 246576 | 227744 |
| 0.1/3 | 3754912 | 1450832 | 202288 | 178784 |

All ten comparison launches complete 648 model calls, 48 local nodes, and all
155136 declared output tokens. Retries, prompt halvings and terminal
preemptions are zero. Target transfer failures and running growth deferrals
are zero. No important request waits at least 180
seconds before admission in these completed runs; none remains unfinished.
Generated downstream context can vary under the same construction rules.
Observed prompt totals for target QPS 1.0/0.5 and QPS 0.1 runs 1/2/3 are
5877697, 5870893, 5878149, 5879268 and 5885200 tokens respectively.

## Verification

Six full-attention capacity rejection reproductions failed before the
precheck. After the change, 11 focused cases pass in 4.01 seconds, including
multiple cache groups, hybrid fallback, shared GPU prefixes, and full request
completion after capacity frees and a 192-token CPU prefix is restored.
The focused JUnit artifact is `.venv/optimization-capacity-probe-focused.xml`.

The CPU regression suite passes 385 tests in 88.87 seconds; three GPU
configuration cases are deselected:

```bash
env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
  tests/tokencake/test_scheduling.py tests/tokencake/test_offloading.py \
  tests/tokencake/test_config.py \
  tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py \
  tests/v1/kv_offload/cpu/test_manager.py -q -k 'not data_parallel_rejected' \
  --junitxml=.venv/optimization-capacity-probe-cpu-unit.xml
```

Four actual-model scheduling and CPU restoration tests pass in 174.04
seconds, before the formal E2E campaign starts:

```bash
env PATH="/root/TokenCake/vLLM-TokenCake-upstream/.venv/bin:$PATH" \
  HF_HUB_OFFLINE=1 \
  VLLM_TEST_MODEL=/root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct \
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
  tests/tokencake/test_scheduling_serving.py \
  tests/tokencake/test_offloading_serving.py -q \
  --junitxml=.venv/optimization-capacity-probe-model.xml
```

Changed-file pre-commit hooks, including type checks, pass. The formal run is:

```bash
.venv/bin/python -m tools.tokencake_experiments.driver run-cases \
  /root/autodl-tmp/tokencake-optimization/conversation-tools-02
```

The affected QPS 0.1 pair is then completed with the existing
`Runner.repeat_affected(include_old=False)` under the original campaign lock.
Both extra launches per member use the original frozen identity and budget.
Frozen-input, runtime and environment checks remain enabled. The final GPU
process check is empty, with both experiment drivers exited successfully.

An isolated CPU diagnostic uses the real scheduler and CPU cache metadata,
with 10 running and 62 waiting requests, and no model execution or GPU copies.
It reduces unprofiled scheduler time from 12.901 to 2.021 ms per step, about
84%. This is not an E2E result. The full QPS 1.0 runtime is 660.10 seconds
versus the preceding stage's 656.55 seconds, so the capacity precheck alone
has not demonstrated an additional E2E reduction. The formal comparison
evaluates the integrated implementation against native on the revised
workload; it does not isolate contributions from the individual policies.
