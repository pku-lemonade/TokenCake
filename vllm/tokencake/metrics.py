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


class TokenCakeMetrics:
    def __init__(self) -> None:
        self._counters = dict.fromkeys(Metric, 0)

    def count(self, metric: Metric, amount: int = 1) -> None:
        assert amount >= 0
        self._counters[metric] += amount

    def snapshot(self, active: int) -> dict[str, int]:
        return {metric.value: value for metric, value in self._counters.items()} | {
            "active": active
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

    def observe(self, values: dict[str, int], engine_idx: int) -> None:
        previous = self._previous[engine_idx]
        for metric in Metric:
            value = values[metric.value]
            delta = value - previous.get(metric.value, 0)
            assert delta >= 0
            self._counters[engine_idx, metric.value].inc(delta)
        self._active[engine_idx].set(values["active"])
        self._previous[engine_idx] = values
