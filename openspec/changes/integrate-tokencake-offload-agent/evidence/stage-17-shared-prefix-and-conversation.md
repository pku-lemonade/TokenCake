# Shared-prefix preservation and conversation workload, 2026-09-06

Completed full-attention prefixes can now be preserved while another request
still references their GPU blocks. The contents are immutable, ownership is
unchanged, and native transfer fences protect later source reuse. Other cache
types still require free source blocks. Three reproductions failed before the
change because the shared prefix was omitted from the preservation batch.

The `conversation` workload appends role instructions after the accumulated
conversation, retains model output across tools with model successors, and
deduplicates common prefixes at joins in stable predecessor-name order. This
extends reusable prefixes to all 26 model-to-model transitions per DAG. Those
tools declare their prefixes eligible; actual selection still checks pressure,
duration and transfer cost. This profile retains tool-call text that older
profiles discarded, increasing downstream context. The 24 DAGs, graph, 648
model calls, 271200 output tokens, tool durations and arrivals are unchanged.
Both implementations receive the same revised construction rules. No
speculative decoding is used.

Campaign: `/root/autodl-tmp/tokencake-optimization/conversation-01`.
Runtime tree:
`6947614d8d5ced5b76dfa1928cc813bb66153747f4e2162f19c906091f157c76`.
Target artifact:
`20d0c274b64e51df1b5a218c6caa1f15ffd66f9b54f6979fe0427d5f5ae47254`.
Observed workload:
`ad3062465acbaa4de1ce508f65ac1bc6bf6818c964d9bc5eb0187ac2c2fbb55b`.
Native case key:
`29b1312450a7d191e121db815f3f808dfcfe5f74906844712f2a562f470c2191`.
Offload-agent case key:
`3d46ca7cdecee9f68dcda066784824f7c8fd11eea572c15bdebbc0e195724a4e`.

Native uses GPU 0; offload-agent explicitly uses GPU 1. Each is an A800 80 GB
with its declared NUMA affinity and 0.50 GPU memory utilization. Both fresh
processes qualify. Result paths are
`cases/phase-1/native/qps-1.0/launch-0/result.json` and
`cases/phase-1/offload-agent/qps-1.0/gpu-1/launch-0/result.json`.

| Metric | Native | Offload-agent |
| --- | ---: | ---: |
| Total E2E seconds | 1504.246461 | 1319.488144 |
| Average application latency seconds | 1361.330853 | 1105.454335 |
| Application p95 seconds | 1472.537853 | 1280.183662 |
| Application p99 seconds | 1475.868126 | 1292.115618 |
| Physical / reservation preemptions | 77 / 0 | 0 / 0 |
| Reservation / capacity denials | N/A | 2721 / 1382382 |
| Executed / recomputed tokens | N/A | 1516239 / 0 |
| GPU / CPU admission hit tokens | N/A | 4241696 / 1711248 |
| Resume GPU / CPU hit tokens | N/A | 0 / 0 |
| Maximum critical admission wait seconds | N/A | 349.554 |
| Critical admissions waiting at least 180 seconds | N/A | 39 |
| Completed saved GPU blocks | N/A | 22839 |
| Scheduler iteration observations | 36099 | 40389 |

Total E2E improves by 12.28% against this profile's fresh native baseline.
The required 25% reduction would need at most 1128.184846 seconds. This stage
does not meet that target. Average application latency and both tails improve,
but substantial critical admission waits remain. More CPU hits alone are not
sufficient: TokenCake still needs more iteration observations than native.

Both cases complete every declared DAG, node, model call and output token.
Retries, prompt halvings, terminal preemptions and transfer failures are zero.
TokenCake's running growth deferrals are zero. Denials count repeated decisions.
Native lacks TokenCake's execution-range and critical-wait instrumentation;
those quantities are unavailable, not zero.

Verification before freezing:

- Shared-prefix reproductions: 3 failed before the fix;
  `.venv/optimization-shared-prefix-red.xml`.
- Offloading CPU regression: 64 passed in 15.41 seconds;
  `.venv/optimization-shared-prefix-cpu-unit.xml`.
- Actual-model restoration and CUDA data checks: 9 passed in 93.68 seconds;
  `.venv/optimization-shared-prefix-model.xml`. These include source overwrites
  after shared owners finish and restoration through native CPU offload.
- Experiment driver, report, launcher and workload checks: 48 passed in
  15.27 seconds; `.venv/optimization-conversation-driver-unit.xml`.
- Changed-file pre-commit hooks, including type checks: passed.

Tests use `.venv/bin/python -m pytest`. CPU tests set `CUDA_VISIBLE_DEVICES=`;
model tests set `CUDA_VISIBLE_DEVICES=1`, `HF_HUB_OFFLINE=1`, and
`VLLM_TEST_MODEL=/root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct`.
The isolated workload probe checks all 27 model calls, all 26 continuation
prefixes, stable join order, original tool times and unchanged output budgets.

An auxiliary replay of the previous continuation workload used the real
synchronous scheduler with recorded generated text and virtual step times.
Expanding the shared-cache score band from 500 to 1500 increased work from
34486 to 34658 steps. This was not adopted. That replay excludes CPU offload,
has one output-retokenization length mismatch, and is not an E2E result.
