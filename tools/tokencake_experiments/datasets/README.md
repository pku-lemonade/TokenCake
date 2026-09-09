# Fixed DAG Datasets

The current workload is [conversation-tools.json](conversation-tools.json).
Each file is self-contained: 24 initial inputs, one shared DAG, role templates,
output budgets, agent annotations, and simulated tool waits and result text.
Normal benchmarks load these files directly without the original workload
builder, tokenizer, repository context extraction, or patch application.

| Field | Contents |
| --- | --- |
| `schema_version` | Validated dataset format version, currently `1` |
| `applications[].initial_input` | Complete fixed initial context |
| `applications[].tool_results` | Fixed result overrides for that application |
| `templates` | Literal role instructions indexed by template name |
| `nodes[].predecessors` | Dependencies in explicit composition order |
| `nodes[].template` | Key into `templates` |
| `nodes[].composition` | How predecessor conversations and the template combine |
| `nodes[].max_tokens` | Fixed generation budget; zero for local nodes |
| `nodes[].metadata` | Agent importance and DAG/offload annotations |
| `nodes[].tool` | Simulated wait, literal result, and generation retention rule |
| `max_depth` | Original graph depth used by the scheduling metadata |
| `provenance` | Source revision, seed, and export notes |

For the conversation profiles, a linear successor receives:

```text
predecessor input
+ predecessor text generated in this run
+ fixed tool result, when present
+ successor template
```

Joins wait for every declared predecessor. They retain the common chunk prefix
once and append the branch suffixes in the JSON predecessor order, followed by
the next template. Model outputs are never read from this dataset or from a
previous benchmark result.

The four files retain the existing profile names:

| Profile | Composition | Output tokens per DAG |
| --- | --- | ---: |
| `frozen` | Original role prepending and shared-prefix joins | 11300 |
| `continuation` | Same-role validation-to-repair continuation | 11300 |
| `conversation` | Append role instructions after upstream conversation | 11300 |
| `conversation-tools` | Conversation with 128-token tool instructions | 6464 |

Each has 29 executable nodes (27 model calls, two local nodes) and 36 edges.
Tool waits are simulated; the tool result strings do not execute external code.
Search results that were previously random are now fixed samples. Unicode in
the initial inputs is retained from the original source data.
The historical `frozen` and `continuation` profiles merged branches in completion
order; their JSON versions use the declared predecessor order. This removes a
source of run-to-run input variation. The conversation profiles already used
declared order, so their merge behavior is retained.

Run the current dataset against an already running local server:

```bash
.venv/bin/python -m tools.tokencake_experiments.dataset_client \
  --dataset tools/tokencake_experiments/datasets/conversation-tools.json \
  --port 8055 --model_path /root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct \
  --request_rate 1.0 --tokencake-mode native \
  --output_dir /path/to/new/apps --output_file /path/to/new/output_record.json
```

For comparisons, use `driver prepare --workload-profile conversation-tools`
with the desired modes and QPS. An optional `--workload-dataset PATH` selects
an edited file. The campaign still requires all 24 applications; all compared
modes use the same copied JSON bytes, budgets and arrival trace. Runtime
retries retain the existing error policy and disqualify a performance sample.

`export_dataset.py` records the one-time migration from the clean frozen source
revision. It is an audit/migration utility, not part of benchmark preparation or
execution. It never applies patches:

```bash
.venv/bin/python -m tools.tokencake_experiments.export_dataset \
  --source ../vllm_agent \
  --model /root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct \
  --output /path/to/new/datasets
```

Editing the checked-in JSON is sufficient for subsequent workload revisions.
New dataset/client hashes require fresh measurements, including a matching
native baseline. Old performance reports do not certify this client migration.
