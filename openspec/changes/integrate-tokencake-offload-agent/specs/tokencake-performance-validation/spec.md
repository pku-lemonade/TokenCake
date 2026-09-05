## Purpose

Define a reproducible, correctness-gated end-to-end evaluation that determines whether the migrated TokenCake scheduling and offload behavior meets its performance targets on the agreed A800 agent workload.

## ADDED Requirements

### Requirement: Primary workload and runtime settings SHALL be frozen

The primary validation SHALL run 24 complete code-paper-pressure DAG applications from `agentcodeclean_new.json` at offered QPS values `1.0`, `0.5`, and `0.1`, in that high-to-low order for every mode. Every compared mode SHALL use the same A800 model-serving hardware, model, workload records, random seed, arrival schedule, prompt and generation settings, `gpu_memory_utilization=0.50`, and maximum model length `32768`. The declared model SHALL be `/root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct`, and offload-enabled target cases SHALL use the same declared native host-offload capacity, initially 100 GiB.

#### Scenario: A primary comparison is prepared

- **WHEN** the primary end-to-end matrix is generated
- **THEN** it contains one case for every required mode at QPS `1.0`, `0.5`, and `0.1`
- **AND** every case identifies the same frozen set of 24 DAG applications and arrival schedule
- **AND** every case records the common model and memory settings

#### Scenario: A workload input changes

- **WHEN** a candidate run uses a different DAG record, seed, arrival offset, prompt, generation setting, model, or applicable memory setting from its comparison peer
- **THEN** the run MUST be marked non-comparable and MUST NOT contribute to a hard performance gate

### Requirement: Validation modes SHALL isolate the migrated mechanisms

The required target modes SHALL be native v0.22 baseline, migrated agent-only, and migrated offload-agent. Native baseline SHALL use the default v0.22 scheduler with no TokenCake configuration, controller, or KV offload. Agent-only SHALL enable TokenCake scheduling while disabling TokenCake offload and native CPU offload. Offload-agent SHALL enable TokenCake scheduling and TokenCake's extension of v0.22 native CPU offload. The latest old offload-agent and current Mooncake cases SHALL be comparison references rather than substitutions for any target mode.

#### Scenario: Native baseline is launched

- **WHEN** the native v0.22 baseline case starts
- **THEN** it uses the unmodified native scheduling and GPU-only KV-cache path
- **AND** no TokenCake controller or offload connector is active

#### Scenario: Mechanism isolation is checked

- **WHEN** target agent-only and target offload-agent results are compared
- **THEN** their frozen workload and serving settings SHALL differ only in the offload-related configuration required by those modes

### Requirement: Every benchmark case SHALL use a fresh server

Each benchmark case SHALL start a fresh server process, wait until that server is ready, run exactly the assigned workload, collect its outputs, metrics, and logs, and terminate it before the next case on that GPU. Server startup time SHALL be excluded from the primary performance interval.

#### Scenario: Consecutive QPS cases run

- **WHEN** a mode finishes its QPS `1.0` case and proceeds to QPS `0.5`
- **THEN** the QPS `0.5` case SHALL use a newly started server with no cache, lifecycle, prediction, or backoff state inherited from QPS `1.0`

### Requirement: The two A800 execution queues SHALL run concurrently with fixed affinity

The matrix SHALL use one server per A800 and SHALL run two independent, concurrent queues. GPU0 (`GPU-dfe6761a-23f4-c34c-9c03-01da02020e5f`) SHALL be bound to NUMA node 0 CPUs `0-35,72-107` and execute native baseline at QPS `1.0`, `0.5`, `0.1`, followed by target offload-agent at QPS `1.0`, `0.5`, `0.1`. GPU1 (`GPU-ce81ab13-d8a0-a49d-c908-f876b8eb2087`) SHALL be bound to NUMA node 1 CPUs `36-71,108-143` and execute target agent-only at QPS `1.0`, `0.5`, `0.1`. After all three target modes finish, GPU0 SHALL execute latest-old offload-agent and GPU1 SHALL execute Mooncake, each at QPS `1.0`, `0.5`, `0.1`. The server and its benchmark client SHALL share the assigned affinity. Under the accepted single-A800 replacement rule, these fixed A800/NUMA assignments SHALL be treated as directly comparable; actual peer or idle state SHALL be recorded, and the workflow SHALL NOT silently add counterbalancing, cooldown, or thermal-invalidating runs.

#### Scenario: Primary target modes run in parallel

- **WHEN** the primary target matrix begins
- **THEN** the GPU0 native-baseline queue and GPU1 agent-only queue SHALL be allowed to execute concurrently
- **AND** each queue SHALL preserve its declared within-GPU order

#### Scenario: A case is traced to concurrent activity

- **WHEN** a case runs while another controlled benchmark case is active on the peer GPU
- **THEN** its provenance SHALL record both GPU UUIDs, both CPU affinities, and the identity of the concurrent peer case

### Requirement: Total end-to-end time SHALL be the primary metric

The primary metric `performance.total_e2e_s` SHALL measure wall-clock time from launching the benchmark client subprocess until that subprocess exits after all 24 DAG applications complete. It SHALL exclude server startup. Average, P50, P90, P95, P99, maximum application latency, throughput, retry counts, prompt-truncation counts, and available queue, tool, transfer, and residual timing SHALL be retained as diagnostics but SHALL NOT replace `total_e2e_s` in a hard gate.

#### Scenario: Primary duration is calculated

- **WHEN** all 24 applications have completed and the benchmark client exits
- **THEN** `performance.total_e2e_s` SHALL equal the recorded client-process wall-clock interval
- **AND** server initialization time SHALL not be included

### Requirement: Offload-agent SHALL satisfy all hard performance gates

For each required QPS, let `T_native`, `T_agent`, `T_target_oa`, and `T_old_oa` be qualifying `performance.total_e2e_s` values for native baseline, target agent-only, target offload-agent, and latest-old offload-agent. Target offload-agent SHALL satisfy `(T_native - T_target_oa) / T_native >= 0.10`; SHALL satisfy `T_target_oa <= 1.05 * T_agent`; and, when the old result is work-equivalent, SHALL satisfy `T_target_oa <= T_old_oa`. These gates SHALL be evaluated independently at every QPS, including the five-percent agent-only bound at QPS `1.0`.

#### Scenario: A single-run point is clearly above every threshold

- **WHEN** all relevant initial results at a QPS are qualifying and each normalized result is more than three percentage points inside its passing region
- **THEN** the strict hard gates SHALL be evaluated from those initial results without routinely repeating the entire point

#### Scenario: Native improvement is insufficient

- **WHEN** the qualifying target offload-agent time does not improve upon native baseline by at least ten percent at a required QPS
- **THEN** the native-improvement hard gate SHALL fail for that QPS after applying the repetition rule

#### Scenario: Agent-only regression exceeds the allowance

- **WHEN** qualifying target offload-agent time is more than five percent slower than target agent-only at a required QPS
- **THEN** the agent-only-relative hard gate SHALL fail for that QPS after applying the repetition rule

### Requirement: Repetitions SHALL be limited to affected comparisons

Every case SHALL initially run once. If an initial hard-gate comparison fails its threshold or its normalized result lies within three percentage points on either side of that threshold, only the cases in that relevant comparison pair SHALL be brought to three total launches. Measurements already collected for the same mode, QPS, and implementation identity by another hard-gate pair SHALL be reused; a shared case SHALL never be launched more than three times, including invalid or contaminated launches. A repeated comparison SHALL use the median qualifying `total_e2e_s` for each side and SHALL apply the original strict threshold to those medians. If exclusions exhaust the three-launch budget before a qualifying comparison exists, the gate SHALL remain unresolved and require explicit direction. The workflow MUST NOT run every matrix case three times by default.

For target offload-agent, the implementation phase and target artifact hash SHALL be part of the case identity. If evidence triggers a Phase-2 code change, Phase-1 target-offload-agent measurements SHALL NOT be pooled with Phase 2, and each affected Phase-2 mode/QPS SHALL receive a fresh budget of at most three launches. A qualifying native, agent-only, or old reference measurement MAY be reused only when its executable behavior, configuration, workload, and environment relevant to that comparison are unchanged and the equivalence is recorded; otherwise that comparison member SHALL run under its own applicable budget.

#### Scenario: A result enters the gray band

- **WHEN** an initial normalized result is within `0.03` of its applicable hard-gate boundary
- **THEN** each mode forming that gate SHALL be brought to at most three total launches for that QPS, reusing any measurements already collected for another gate
- **AND** the gate SHALL be decided from their medians using the strict boundary

#### Scenario: A shared target case already has three launches

- **WHEN** target offload-agent was already launched three times for one hard gate and another comparison at the same QPS triggers repetition
- **THEN** its qualifying measurements SHALL be reused and only the other comparison member MAY be brought up to its three-launch limit

#### Scenario: One pair requires repetition

- **WHEN** the native-versus-target-offload-agent comparison triggers repetition but the target-agent-only comparison is clearly passing outside its gray band
- **THEN** only native baseline and target offload-agent SHALL be repeated for that decision
- **AND** target agent-only SHALL not be repeated solely because another gate was close

#### Scenario: Phase 2 changes target offload-agent

- **WHEN** the evidence gate authorizes a Phase-2 implementation and an affected comparison is rerun
- **THEN** Phase-2 target offload-agent starts a separate at-most-three-launch budget for that QPS
- **AND** Phase-1 target offload-agent timings are retained as provenance but are not included in the Phase-2 median
- **AND** an unchanged qualifying comparison member MAY be reused only with recorded equivalence

### Requirement: Latest-old parity SHALL require work equivalence

The initial old-code comparison SHALL run only latest-old offload-agent at the three required QPS values, using the unchanged source behavior on the same A800 and frozen workload. Its result SHALL be eligible for the hard old-parity gate only when the run reports zero `FINISHED_PREEMPTED` requests. A result with one or more `FINISHED_PREEMPTED` requests SHALL be retained and reported as work-inequivalent, SHALL NOT be used to claim that the migrated target failed old-code parity, and SHALL require explicit follow-up direction before an alternative parity comparison is introduced. Old agent-only or additional old-code diagnostics SHALL run only when needed to investigate a target miss against a qualifying old offload-agent result.

#### Scenario: Old comparison performs equivalent work

- **WHEN** a latest-old offload-agent run reports `FINISHED_PREEMPTED=0` and passes all correctness checks
- **THEN** its `total_e2e_s` SHALL participate in the target-versus-old hard gate at the matching QPS

#### Scenario: Old comparison terminates preempted work

- **WHEN** a latest-old offload-agent run reports `FINISHED_PREEMPTED>0`
- **THEN** the result SHALL be preserved with a work-inequivalent designation
- **AND** it SHALL be excluded from the hard old-parity decision

### Requirement: Mooncake SHALL be a correctness-qualified speed reference only

The current `mooncake_agent` environment and tokencake-mooncake configuration SHALL run the same model, frozen workload, and QPS values once each after the target primary cases complete on GPU1. Mooncake SHALL pass workload correctness and connector-error checks. Its timing SHALL be reported as a speed reference and SHALL NOT create a hard speed gate for the migrated implementation. An excluded Mooncake launch SHALL be reported without an automatic replacement; another launch requires explicit direction.

#### Scenario: Mooncake is slower than target offload-agent

- **WHEN** a correctness-qualified Mooncake case has a larger `total_e2e_s` than target offload-agent
- **THEN** the difference SHALL be reported
- **AND** no additional hard gate SHALL be inferred from it

#### Scenario: Mooncake connector fails

- **WHEN** the Mooncake server or connector reports an execution error
- **THEN** the case SHALL be marked non-qualifying even if the client produces a timing value

### Requirement: End-to-end correctness SHALL qualify every performance result

A result SHALL qualify for performance comparison only when the full end-to-end launcher completes under its maintained analyzer semantics, all expected applications and nodes are accounted for, the server and active connector report no fatal execution error, and every required TokenCake event request succeeds. Unit tests MAY support implementation confidence but SHALL NOT satisfy the end-to-end acceptance criteria. Smoke tests and smoke-truncated application runs MUST NOT be used for acceptance.

#### Scenario: A smoke run completes quickly

- **WHEN** a run stops after a subset of application nodes or uses any smoke-limit option
- **THEN** its timing SHALL be excluded from every acceptance gate

#### Scenario: A lifecycle event returns an error

- **WHEN** the adapted launcher receives a non-success HTTP status for `stall_started` or `stall_finished`
- **THEN** it SHALL fail the benchmark case instead of silently accepting the response body

#### Scenario: Supporting unit tests pass without an end-to-end run

- **WHEN** all selected unit tests pass but no qualifying 24-DAG case has run
- **THEN** performance acceptance SHALL remain undetermined

### Requirement: Launcher adaptation SHALL preserve the latest workload semantics

Target runs SHALL use a small target-owned protocol patch applied to a disposable checkout of the frozen latest `../vllm_agent` source. The source checkout itself SHALL remain unchanged, and latest-old comparisons SHALL retain the old launcher protocol. The target patch SHALL be limited to target request construction, UUID lifecycle correlation, generic TokenCake event submission, and HTTP error propagation. It SHALL preserve the old launcher's DAG execution, branch and arrival behavior, timing boundaries, analyzer behavior, retry loop (`retry_count <= 3`, permitting up to four attempts), prompt halving after a failed attempt, and existing retry, token, and output-text acceptance semantics.

#### Scenario: A target LLM attempt is constructed

- **WHEN** the adapted launcher starts any LLM attempt
- **THEN** it SHALL generate a fresh UUID4 attempt ID formatted `tc-<uuidhex>` and use it as the top-level request ID
- **AND** agent-only and offload-agent SHALL place the same value in nested TokenCake `lifecycle_id`
- **AND** offload-agent SHALL use that value for all lifecycle events, while agent-only SHALL send no events and native baseline SHALL send neither nested TokenCake metadata nor events
- **AND** application, node, and attempt trace fields SHALL be recorded separately rather than used as the uniqueness mechanism

#### Scenario: A request is retried

- **WHEN** an LLM attempt fails and the maintained retry loop starts another attempt
- **THEN** the launcher SHALL halve the prompt according to the old behavior
- **AND** it SHALL assign the retry a new top-level UUID4 attempt ID and, for an annotated mode, the same new nested lifecycle ID
- **AND** it SHALL NOT add a new retry-count, token-count, or output-text hard gate

#### Scenario: A disposable launcher is prepared

- **WHEN** a target benchmark launcher is materialized
- **THEN** it SHALL be derived from the frozen source commit plus the recorded target-owned patch
- **AND** the original `../vllm_agent` worktree SHALL remain unchanged

### Requirement: Every result SHALL carry reproducible provenance

Each case SHALL persist its command, full resolved configuration, mode, QPS, implementation phase, runtime artifact hash, run identifier, timestamps, source commit, target commit, launcher source commit, launcher patch hash, dataset path and checksum, model identity, environment identity, relevant package or wheel versions and checksums, GPU UUID, CPU and NUMA affinity, random seed, arrival trace identity, concurrent peer case, raw launcher output, server logs, collected metrics, correctness status, exclusion reason, and final gate inputs. The source workload SHALL be identified as `/root/TokenCake/vllm_agent/dataset/agentcodeclean_new.json` with checksum `730f251343121339d87313895add089f2520971319bdc92c282d5497a6425ca0`. Provenance SHALL distinguish the current Mooncake runtime package from the tokencake-mooncake configuration source.

#### Scenario: A result is reviewed later

- **WHEN** a reviewer opens a persisted result without access to the original process state
- **THEN** the recorded provenance SHALL be sufficient to identify the exact code, launcher transformation, workload, runtime, hardware assignment, peer load, and command that produced it

#### Scenario: A run is excluded

- **WHEN** a run cannot contribute to a gate because of correctness, contamination, work-equivalence, or comparability rules
- **THEN** its available raw artifacts SHALL still be retained
- **AND** its explicit exclusion reason SHALL be recorded

### Requirement: External GPU contamination SHALL invalidate a run

The benchmark SHALL monitor the assigned GPU for processes outside the two controlled benchmark queues. Detection of an unrelated process using the assigned GPU during a case SHALL invalidate that case and trigger a replacement only while the mode/QPS remains below its three-launch budget; Mooncake's once-per-QPS reference is not automatically replaced. A declared benchmark running concurrently on the other assigned GPU SHALL NOT be treated as third-party contamination.

#### Scenario: An unrelated GPU process appears

- **WHEN** a process not owned by the declared benchmark queues uses the case's assigned GPU during its measured interval
- **THEN** that measurement SHALL be marked invalid
- **AND** the process evidence and rerun disposition SHALL be recorded

#### Scenario: Contamination exhausts the launch budget

- **WHEN** a hard-gate mode/QPS has already launched three times and exclusions leave insufficient qualifying measurements
- **THEN** no fourth launch is scheduled automatically
- **AND** the affected gate remains unresolved pending explicit direction

#### Scenario: The controlled peer queue is active

- **WHEN** the declared peer benchmark runs on the other A800 during a case
- **THEN** the case SHALL remain eligible with the peer recorded in provenance

### Requirement: Thermal state SHALL be diagnostic only

GPU temperature, power, and clock observations SHALL be recorded when available, but they SHALL NOT impose cooldowns, invalidate a run, or automatically trigger a rerun in this acceptance round.

#### Scenario: Temperature or clock varies

- **WHEN** a case observes a different temperature, power level, or clock from its comparison peer without third-party GPU contamination
- **THEN** the observation SHALL be reported as diagnostic context
- **AND** the run SHALL remain eligible under the thermal rule
