# Stage 3 Validation

The lifecycle/event foundation is implemented: generation association,
independent completion/start/terminal facts, detached per-group snapshots,
lazy expiry, bounded duration history, exact ready-key reference accounting,
authenticated events, typed core acknowledgements, reset coordination, and
fixed-cardinality metrics. Annotated generation-time stores are deferred
before native store progress advances.

Snapshot policy selection and detached asynchronous job publication remain
for the native-offload stage. Tasks 3.1, 3.4, and 3.6 remain open until their
transfer-dependent and component-refusal scenarios are also verified.

## Real Model Checks

The following model-backed suites ran concurrently on the frozen GPU0 and
GPU1 devices respectively. Each used Qwen2.5-14B-Instruct's original eight
shards, TP=PP=DP=1, eager execution, 4096 context, 0.5 GPU memory utilization,
and a fresh server per configuration. Both server trees shut down afterward.

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=0 VLLM_TEST_MODEL=/root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct .venv/bin/python -m pytest tests/tokencake/test_event_serving.py -v -s -x --junitxml=openspec/changes/integrate-tokencake-offload-agent/evidence/stage-3-event-model-final.xml
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=1 VLLM_TEST_MODEL=/root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct .venv/bin/python -m pytest tests/tokencake/test_serving.py -v -s -x --junitxml=openspec/changes/integrate-tokencake-offload-agent/evidence/stage-3-http-regression.xml
```

Results: **8 passed** in 38.01 seconds and **32 passed** in 128.62 seconds.
The event suite exercises all three generation APIs, start/finish replay,
conflict, native duplicate-generation error output while the original
generation continues, start-before-completion, early finish followed by
completion or disconnect, abort tombstones, API-key authentication, unknown
IDs, excluded legacy routes, local/external native reset, and Prometheus
labels. An early test attempt closed its streaming iterator before posting
start; keeping the iterator alive corrected the test ordering.

These functional checks do not qualify the full 24-DAG acceptance workload
or consume its performance launch budget. Native D2H job correctness and
performance gates still require the later offload and acceptance stages.

## Supporting Checks

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/tokencake/test_events.py tests/tokencake/test_lifecycle.py -q --tb=short --junitxml=openspec/changes/integrate-tokencake-offload-agent/evidence/stage-3-lifecycle-events.xml
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/tokencake/test_config.py tests/tokencake/test_protocol.py tests/v1/kv_connector/unit/offloading_connector -q
```

Results: **41 passed** and **214 passed**. Tests cover deterministic expiry
boundaries, first-terminal authority, four-case prediction, the 4096-entry
history bound, anonymous-history collection, LRU/ARC ready references,
stacked native loads and lifecycle references, and release/reset before
store completion. Endpoint checks also verify that dispatch alone cannot
complete the HTTP response and that internal exception text is not exposed.

Changed-file pre-commit checks include mypy. JUnit artifacts accompany this
report; local full logs use the corresponding `.log` basenames. This is an
AI-assisted intermediate implementation and requires human review before
upstream submission.
