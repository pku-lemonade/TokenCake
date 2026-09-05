# Stage 5: Native CPU Preservation

Implemented snapshot-only Phase-1 preservation through the native CPU manager,
shared transfer-job counter, worker metadata and source-reuse fences. Annotated
generation stores stay deferred before their progress index advances. Ordinary
store/load work still delegates to the native connector.

Every detached job fences all source blocks at registration and first appears
in a zero-model-token publication step. The next native worker step submits D2H
before source-reuse waits or model writes. Completion uses detached context,
never a finished Request. Ready CPU ownership is retained once per lifecycle;
pending stores retain only after native completion and while ownership remains
active. Finishing, expiry and reset cannot resurrect a released reference.

The policy uses native multi-group waiting allocation demand, the source's
pressure/utility/cost/temporal/backoff rules, and one bounded preservation batch
per lifecycle. Multi-group truncation requires a common native hit boundary and
complete applicable sliding windows. Physical GPU capacity does not increase.
The bounded free-LRU observation only supplies attribution counters.

Local and external reset follow native component ordering. Unpublished jobs
never enter a worker wait; published jobs retain native reset fencing. Duration
history, transfer bandwidth EWMAs and monotonic metrics survive external reset.
Native post-ack submission/completion failures remain fail-closed.

## Execution Evidence

| Validation | Result | Artifact |
| --- | --- | --- |
| Qwen2.5-14B, offload-only and offload-agent: preserved generation, 12 concurrent eviction requests, GPU reset, positive native H2D bytes, identical recovered output, external reset and recomputation | 2 passed, 90.98 s | `stage-5-model-offload.xml` |
| Qwen2.5-14B, three APIs, four configuration modes, streaming, authenticated lifecycle events, abort, duplicate generation and reset | 40 passed, 162.59 s | `stage-5-model-api-regression.xml` |
| Native CUDA worker, single group, larger CPU blocks, different full-attention block sizes and hybrid sliding windows: real D2H, fenced source overwrite, native prefix lookup/H2D, every retained block compared byte-for-byte | 4 passed, 2.84 s | `stage-5-cuda-final.xml` |
| Scheduler/CPU manager state, native worker ordering, reference stacking, policy, reset and failures, plus CUDA transfers | 51 passed, 13.41 s | `stage-5-offloading-regression.xml` |
| Configuration, protocol, lifecycle, events, scheduling, original connector and CPU manager | 288 passed, 69.36 s | `stage-5-native-regression.xml` |

GPU commands used `PATH="$PWD/.venv/bin:$PATH"` and
`VLLM_TEST_MODEL=/root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct`:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
  tests/tokencake/test_offloading_serving.py -x -q \
  --junitxml=openspec/changes/integrate-tokencake-offload-agent/evidence/stage-5-model-offload.xml
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
  tests/tokencake/test_serving.py tests/tokencake/test_event_serving.py -x -q \
  --junitxml=openspec/changes/integrate-tokencake-offload-agent/evidence/stage-5-model-api-regression.xml
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/tokencake/test_offloading.py tests/tokencake/test_offloading_cuda.py -x -q \
  --junitxml=openspec/changes/integrate-tokencake-offload-agent/evidence/stage-5-offloading-regression.xml
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/tokencake/test_offloading_cuda.py -x -q \
  --junitxml=openspec/changes/integrate-tokencake-offload-agent/evidence/stage-5-cuda-final.xml
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
  tests/tokencake/test_config.py tests/tokencake/test_protocol.py \
  tests/tokencake/test_lifecycle.py tests/tokencake/test_events.py \
  tests/tokencake/test_scheduling.py \
  tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py \
  tests/v1/kv_offload/cpu/test_manager.py -x -q \
  --junitxml=openspec/changes/integrate-tokencake-offload-agent/evidence/stage-5-native-regression.xml
```

Changed-file pre-commit checks passed, including Ruff and mypy. Source git status
is clean. Code search found no old swap maps, related-block graph, upload debt or
reservation machinery, terminal preemption status, or legacy MCP paths in the
TokenCake runtime package. An initial native regression command used the wrong
CPU-manager test filename and collected no tests; its XML is retained locally
at `.venv/stage-5-invalid-test-path.xml`, followed by the successful command above.

Unit tests are supporting evidence. Actual CUDA data and model execution verify
this stage's behavior; the specified complete 24-DAG performance gates remain
pending. Multi-group CUDA transfer tests do not establish model-level TP/PP or
hybrid-architecture performance. This implementation used AI assistance and
requires human review before an upstream submission.
