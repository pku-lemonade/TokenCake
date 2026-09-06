# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-cardinality TokenCake counters transported with scheduler statistics."""

from enum import Enum

import prometheus_client


class Metric(str, Enum):
    ASSOCIATED = "lifecycle.associated"
    COMPLETED = "lifecycle.completed"
    STARTED = "lifecycle.started"
    FINISHED = "lifecycle.finished"
    ABORTED = "lifecycle.aborted"
    ERROR = "lifecycle.error"
    EXPIRED = "lifecycle.expired"
    RESET = "lifecycle.reset"
    DUPLICATE = "lifecycle.duplicate"
    LATE_FINISH = "lifecycle.late_finish"
    UNKNOWN = "lifecycle.unknown"
    CONFLICT = "lifecycle.conflict"
    NOT_ELIGIBLE = "decision.not_eligible"
    EMPTY = "decision.empty"
    LOW_PRESSURE = "decision.low_pressure"
    NO_WAITING_DEMAND = "decision.no_waiting_demand"
    BACKOFF = "decision.backoff"
    UNPROFITABLE = "decision.unprofitable"
    PREFIX_GAP = "decision.prefix_gap"
    CPU_CAPACITY = "decision.cpu_capacity"
    SELECTED = "decision.selected"
    STALE = "decision.stale_snapshot"
    EXTERNAL = "decision.snapshot_external"
    FENCE_WAIT = "decision.fence_wait"
    TRANSFER_FAILURE = "decision.transfer_failure"
    SAVED = "saved_blocks.completed"
    DEFERRED = "scheduling.deferred"
    PREEMPTED = "scheduling.preempted"
    PREFILL_CAPPED = "scheduling.prefill_capped"
    PHYSICAL_PREEMPTED = "scheduling.physical_preempted"
    RESERVATION_PREEMPTED = "scheduling.reservation_preempted"
    RESERVATION_DENIED = "scheduling.reservation_denied"
    PREFILL_CAPACITY_DENIED = "scheduling.prefill_capacity_denied"
    GENERATION_CAPACITY_DENIED = "scheduling.generation_capacity_denied"
    RESUME_DEFERRED = "scheduling.resume_deferred"
    ADMITTED = "scheduling.admitted"
    CRITICAL_ADMITTED = "scheduling.critical_admitted"
    CRITICAL_WAIT_GE_60S = "scheduling.critical_wait_ge_60s"
    CRITICAL_WAIT_GE_180S = "scheduling.critical_wait_ge_180s"
    EXECUTED_TOKENS = "scheduling.executed_tokens"
    RECOMPUTED_TOKENS = "scheduling.recomputed_tokens"
    GPU_HIT_TOKENS = "scheduling.gpu_hit_tokens"
    CPU_HIT_TOKENS = "scheduling.cpu_hit_tokens"
    RESUME_GPU_HIT_TOKENS = "scheduling.resume_gpu_hit_tokens"
    RESUME_CPU_HIT_TOKENS = "scheduling.resume_cpu_hit_tokens"


class TokenCakeMetrics:
    def __init__(self) -> None:
        self._counters = dict.fromkeys(Metric, 0)
        self._max_critical_wait_ms = 0

    def admission_wait(self, seconds: float, *, critical: bool) -> None:
        self.count(Metric.ADMITTED)
        if critical:
            self.count(Metric.CRITICAL_ADMITTED)
            self._max_critical_wait_ms = max(
                self._max_critical_wait_ms, int(seconds * 1000)
            )
            if seconds >= 60:
                self.count(Metric.CRITICAL_WAIT_GE_60S)
            if seconds >= 180:
                self.count(Metric.CRITICAL_WAIT_GE_180S)

    def count(self, metric: Metric, amount: int = 1) -> None:
        assert amount >= 0
        self._counters[metric] += amount

    def snapshot(self, active: int) -> dict[str, int]:
        return {metric.value: value for metric, value in self._counters.items()} | {
            "active": active,
            "max_critical_wait_ms": self._max_critical_wait_ms,
        }


class TokenCakeProm:
    def __init__(self, engine_indexes: list[int]) -> None:
        self._previous: dict[int, dict[str, int]] = {i: {} for i in engine_indexes}
        counters = {
            group: prometheus_client.Counter(
                f"vllm:tokencake_{group}_total",
                f"TokenCake {group.replace('_', ' ')} outcomes.",
                labelnames=["engine", "outcome"],
            )
            for group in ("lifecycle", "decision", "saved_blocks", "scheduling")
        }
        self._counters = {
            (i, metric.value): counters[metric.value.split(".")[0]].labels(
                str(i), metric.value.split(".")[1]
            )
            for i in engine_indexes
            for metric in Metric
        }
        active = prometheus_client.Gauge(
            "vllm:tokencake_active_lifecycles",
            "TokenCake lifecycles awaiting completion, start, or finish.",
            labelnames=["engine"],
            multiprocess_mode="mostrecent",
        )
        self._active = {i: active.labels(str(i)) for i in engine_indexes}
        wait = prometheus_client.Gauge(
            "vllm:tokencake_critical_queue_wait_max_seconds",
            "Maximum observed critical request admission wait since server start.",
            labelnames=["engine"],
            multiprocess_mode="mostrecent",
        )
        self._wait = {i: wait.labels(str(i)) for i in engine_indexes}

    def observe(self, values: dict[str, int], engine_idx: int) -> None:
        previous = self._previous[engine_idx]
        for metric in Metric:
            value = values[metric.value]
            delta = value - previous.get(metric.value, 0)
            assert delta >= 0
            self._counters[engine_idx, metric.value].inc(delta)
        self._active[engine_idx].set(values["active"])
        self._wait[engine_idx].set(values.get("max_critical_wait_ms", 0) / 1000)
        self._previous[engine_idx] = values
