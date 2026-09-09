# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-threaded lifecycle coordination owned by the native scheduler."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from vllm.tokencake.config import OffloadConfig
from vllm.tokencake.events import Disposition, LifecycleEvent, LifecycleEventResult
from vllm.tokencake.metrics import Metric, TokenCakeMetrics
from vllm.tokencake.protocol import TokenCakeMetadata

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId
    from vllm.v1.kv_offload.base import OffloadKey, ReqContext
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

_ATTACH_SECONDS = 60.0
_TOMBSTONE_SECONDS = 60.0
_HISTORY_LIMIT = 4096
GenerationCause = Literal["completed", "aborted", "error"]
TerminalCause = Literal["completed", "finished", "aborted", "error", "expired", "reset"]


class DuplicateLifecycleError(ValueError):
    """A second native generation attempted to claim an existing lifecycle."""


@dataclass(frozen=True)
class SnapshotGroup:
    keys: tuple[OffloadKey, ...]
    block_ids: tuple[tuple[int, ...], ...]
    gpu_hashes: tuple[tuple[BlockHashWithGroupId | None, ...], ...]

    def is_valid(self, index: int, pool: BlockPool) -> bool:
        return all(
            bid != 0
            and expected is not None
            and pool.blocks[bid].block_hash == expected
            and pool.blocks[bid].ref_cnt == 0
            for bid, expected in zip(self.block_ids[index], self.gpu_hashes[index])
        )


@dataclass(frozen=True)
class PrefixSnapshot:
    groups: tuple[SnapshotGroup, ...]
    req_context: ReqContext


@dataclass
class Lifecycle:
    metadata: TokenCakeMetadata
    completed_at: float | None = None
    snapshot: PrefixSnapshot | None = None
    start: LifecycleEvent | None = None
    started_at: float | None = None
    predicted_duration: float = 0.0
    safety_deadline: float = float("inf")
    release_deadline: float = float("inf")
    ownership_released: bool = False
    retained: set[OffloadKey] = field(default_factory=set)
    pending_retention: set[OffloadKey] = field(default_factory=set)
    pending_evaluation: bool = False
    next_evaluation_step: int = 0
    backoff_signature: tuple[int, ...] | None = None
    last_store_step: int = -5
    terminal_cause: TerminalCause | None = None
    terminal_at: float | None = None
    finish_accepted: bool = False
    anonymous_sample: float | None = None

    @property
    def state(self) -> str:
        if self.terminal_cause is not None:
            return self.terminal_cause
        if self.start is not None:
            return "released" if self.ownership_released else "active"
        return "generating" if self.completed_at is None else "awaiting_start"

    @property
    def owns_preservation(self) -> bool:
        return (
            self.start is not None
            and self.terminal_at is None
            and not self.ownership_released
        )


class LifecycleRegistry:
    def __init__(
        self,
        settings: OffloadConfig,
        manager: CPUOffloadingManager | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.manager = manager
        self.clock = clock
        self.records: dict[str, Lifecycle] = {}
        # A live generation keeps its association even if its stall tombstone expires.
        self._generations: dict[str, str] = {}
        self._history: OrderedDict[tuple[str, str, str], float] = OrderedDict()
        self.metrics = TokenCakeMetrics()

    def associate(self, request_id: str, metadata: TokenCakeMetadata) -> None:
        self.expire()
        identifier = metadata.lifecycle_id
        if identifier in self.records or identifier in self._generations:
            self.metrics.count(Metric.CONFLICT)
            raise DuplicateLifecycleError(
                "TokenCake lifecycle already belongs to a generation"
            )
        self._generations[identifier] = request_id
        self.records[identifier] = Lifecycle(metadata)
        self.metrics.count(Metric.ASSOCIATED)

    def generation_finished(
        self,
        identifier: str,
        cause: GenerationCause,
        snapshot: PrefixSnapshot | None = None,
    ) -> None:
        self.expire()
        self._generations.pop(identifier, None)
        record = self.records.get(identifier)
        if record is None or record.terminal_at is not None:
            return
        now = self.clock()
        if cause != "completed" or not self.settings.enabled:
            self._terminalize(record, cause, now)
            return
        record.completed_at = now
        record.snapshot = snapshot
        record.pending_evaluation = record.start is not None
        self.metrics.count(Metric.COMPLETED)

    @staticmethod
    def _history_key(record: Lifecycle, kind: str) -> tuple[str, str, str] | None:
        metadata = record.metadata
        if metadata.agent_type:
            return "type", metadata.agent_type, kind
        if metadata.agent_name:
            return "name", metadata.agent_name, kind
        return None

    def _predict(self, record: Lifecycle, event: LifecycleEvent) -> float:
        key = self._history_key(record, event.kind)
        history = self._history.get(key) if key is not None else None
        if key is not None and history is not None:
            self._history.move_to_end(key)
        estimate = event.estimated_duration_s
        if estimate is None:
            return self.settings.default_stall_s if history is None else history
        return estimate if history is None else 0.5 * (estimate + history)

    def _learn(self, record: Lifecycle, now: float) -> None:
        assert record.start is not None and record.started_at is not None
        sample = max(0.0, now - record.started_at)
        key = self._history_key(record, record.start.kind)
        if key is None:
            record.anonymous_sample = sample
            return
        previous = self._history.pop(key, None)
        if previous is not None:
            alpha = self.settings.ewma_alpha
            sample = alpha * sample + (1 - alpha) * previous
        elif len(self._history) >= _HISTORY_LIMIT:
            self._history.popitem(last=False)
        self._history[key] = sample

    def apply(self, event: LifecycleEvent) -> LifecycleEventResult:
        self.expire()
        record = self.records.get(event.lifecycle_id)
        if record is None:
            self.metrics.count(Metric.UNKNOWN)
            return LifecycleEventResult(
                event.lifecycle_id, event.event, "unknown", "unknown", 404
            )
        disposition: Disposition = "applied"
        status = 200
        now = self.clock()
        if event.event == "stall_started":
            if record.start == event:
                disposition = "duplicate"
            elif record.start is not None or record.terminal_at is not None:
                disposition, status = "conflict", 409
            else:
                record.start = event
                record.started_at = now
                record.predicted_duration = self._predict(record, event)
                record.safety_deadline = now + min(
                    3600.0, max(60.0, 4 * record.predicted_duration)
                )
                self.set_h2d_estimate(record, 0.0)
                record.pending_evaluation = record.completed_at is not None
                self.metrics.count(Metric.STARTED)
        elif record.terminal_at is not None:
            disposition = "duplicate" if record.finish_accepted else "late_finish"
        elif record.start is None:
            disposition, status = "conflict", 409
        else:
            self._learn(record, now)
            record.finish_accepted = True
            self._terminalize(record, "finished", now)
        if disposition != "applied":
            self.metrics.count(Metric(f"lifecycle.{disposition}"))
        return LifecycleEventResult(
            event.lifecycle_id, event.event, record.state, disposition, status
        )

    def set_h2d_estimate(self, record: Lifecycle, seconds: float) -> None:
        assert record.started_at is not None
        record.release_deadline = (
            record.started_at
            + record.predicted_duration
            - max(self.settings.release_lead_s, 1.25 * seconds)
        )
        if self.clock() >= record.release_deadline:
            self.release(record)

    def retain_ready(
        self, record: Lifecycle, keys: Iterable[OffloadKey]
    ) -> set[OffloadKey]:
        self.expire()
        if not record.owns_preservation:
            return set()
        assert self.manager is not None
        retained = self.manager.retain(set(keys) - record.retained)
        record.retained.update(retained)
        return retained

    def release(self, record: Lifecycle) -> None:
        keys, record.retained = record.retained, set()
        record.pending_retention.clear()
        record.ownership_released = True
        if keys:
            assert self.manager is not None
            self.manager.release(keys)

    def _terminalize(self, record: Lifecycle, cause: TerminalCause, now: float) -> None:
        if record.terminal_at is not None:
            return
        self.release(record)
        record.snapshot = None
        record.pending_evaluation = False
        record.next_evaluation_step = 0
        record.backoff_signature = None
        record.terminal_cause, record.terminal_at = cause, now
        self.metrics.count(Metric(f"lifecycle.{cause}"))

    def expire(self) -> None:
        now = self.clock()
        for identifier, record in list(self.records.items()):
            if record.terminal_at is not None:
                if now >= record.terminal_at + _TOMBSTONE_SECONDS:
                    del self.records[identifier]
            elif record.started_at is not None:
                if now >= record.safety_deadline:
                    self._terminalize(record, "expired", now)
                elif now >= record.release_deadline:
                    self.release(record)
            elif (
                record.completed_at is not None
                and now >= record.completed_at + _ATTACH_SECONDS
            ):
                self._terminalize(record, "expired", now)

    def invalidate_snapshots(self) -> None:
        for record in self.records.values():
            record.snapshot = None
            record.pending_evaluation = False

    def external_reset_succeeded(self) -> None:
        now = self.clock()
        for record in self.records.values():
            # Native reset already removed these entries regardless of refcount.
            record.retained.clear()
            self._terminalize(record, "reset", now)

    @property
    def has_pending_evaluation(self) -> bool:
        return any(record.pending_evaluation for record in self.records.values())

    def stats(self) -> dict[str, int]:
        return self.metrics.snapshot(
            sum(r.terminal_at is None for r in self.records.values())
        )
