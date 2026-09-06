# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Agent ordering and capacity policy over the native scheduler and block pool."""

import heapq
import math
import time
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import chain
from typing import TYPE_CHECKING

from vllm.tokencake.config import SchedulingConfig
from vllm.tokencake.metrics import Metric, TokenCakeMetrics
from vllm.tokencake.protocol import TokenCakeMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.request_queue import RequestQueue, SchedulingPolicy
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput

_ADJUSTMENT_WINDOW = 500
_HISTORY_LIMIT = 4096
_PREEMPTION_SCORE_BAND = 3000


def request_score(metadata: TokenCakeMetadata, arrival_time: float, now: float) -> int:
    """The reachable latest-source AgentRequestQueue score, with neutral defaults."""
    depth = metadata.depth
    max_depth = max(depth, metadata.application_max_depth, 1)
    remaining = metadata.remaining_depth
    remaining_ratio = min(1.0, remaining / max_depth)
    progress = min(1.0, depth / max_depth)
    elapsed = metadata.application_elapsed_s
    if metadata.application_started_at_s > 0:
        elapsed = max(elapsed, now - metadata.application_started_at_s)
    wait = max(0.0, now - arrival_time)
    similarity = metadata.similarity
    parallel_width = 1 / similarity if 0 < similarity < 1 else 1.0
    age = min(elapsed, 300) * (1 + 0.75 * remaining_ratio)
    queue = min(wait, 180) * (1 + 0.5 * remaining_ratio)
    completion = min(elapsed, 240) * progress
    return int(
        100 * metadata.importance
        + 15 * depth
        + 45 * metadata.out_degree
        + 10 * metadata.in_degree
        + 20 * similarity
        + 20 * remaining
        + 120 * (parallel_width - 1)
        + 6 * min(metadata.application_start_offset_s, 60)
        + 5 * age
        + 8 * queue
        + 5 * completion
        + 180 * metadata.critical_path
        + 90 * metadata.near_completion
        + 20 * bool(metadata.join_group)
        + 4 * (metadata.dependency_depth or depth)
        + 12 * max(metadata.fanout_width - 1, 0)
        - 8 * max(metadata.memory_weight - 1, 0)
    )


@dataclass
class AgentHistory:
    importance: float = 0.0
    deferrals: int = 0
    preemptions: int = 0
    samples: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    completed: int = 0
    duration: float = 0.0
    depth: int = 0
    out_degree: int = 0
    in_degree: int = 0
    elapsed: float = 0.0
    remaining: int = 0

    def score(self, importance: float, waiting: int, average_wait: float) -> float:
        urgency = (
            math.log1p(average_wait)
            + math.log1p(waiting)
            + math.log1p(self.deferrals)
            + 1.5 * math.log1p(self.preemptions)
        )
        tokens = (self.input_tokens + self.output_tokens) / max(1, self.samples)
        duration = self.duration / max(1, self.completed)
        cost = math.log1p(tokens)
        if duration > 0 and tokens > 0:
            cost += math.log1p(duration) + math.log1p(tokens / duration)
        count = max(1, self.samples)
        graph = (
            0.7 * self.depth / count
            + 0.5 * self.out_degree / count
            + 0.25 * self.in_degree / count
            + 0.20 * self.remaining / count
            + 0.10 * min(self.elapsed / count, 180)
        )
        return max(1.0, 2 * importance + urgency + 0.5 * cost + graph)


@dataclass
class CapacityPlan:
    critical: set[str] = field(default_factory=set)
    scores: dict[str, float] = field(default_factory=dict)
    reserved: dict[str, int] = field(default_factory=dict)
    shared: int = 0


def partition_capacity(
    scores: dict[str, float],
    importance: dict[str, float],
    used: dict[str, int],
    total: int,
    reserve_ratio: float,
    critical_ratio: float,
) -> CapacityPlan:
    ordered = sorted(scores, key=lambda k: (scores[k], importance[k], k), reverse=True)
    critical = set(ordered[: max(1, int(len(ordered) * critical_ratio))])
    reserved = int(total * reserve_ratio) if critical else 0
    score_sum = sum(scores[k] for k in critical)
    used_sum = sum(used.values())
    weights = {}
    for key in sorted(critical):
        score_share = scores[key] / score_sum
        memory_share = used.get(key, 0) / used_sum if used_sum else 0.0
        weights[key] = max(score_share, (memory_share + score_share) / 2, 1e-6)
    weight_sum = sum(weights.values())
    exact = {k: reserved * w / weight_sum for k, w in weights.items()}
    counts = {k: int(v) for k, v in exact.items()}
    remainder = reserved - sum(counts.values())
    for key in sorted(exact, key=lambda k: (exact[k] - counts[k], k), reverse=True)[
        :remainder
    ]:
        counts[key] += 1
    return CapacityPlan(critical, scores, counts, total - reserved)


class SchedulingController:
    def __init__(
        self,
        settings: SchedulingConfig,
        manager: KVCacheManager,
        metrics: TokenCakeMetrics,
        num_lookahead_tokens: int = 0,
    ) -> None:
        self.settings = settings
        self.manager = manager
        self.metrics = metrics
        self.num_lookahead_tokens = (
            num_lookahead_tokens if settings.reserve_generation_tokens else 0
        )
        self.metadata: dict[str, TokenCakeMetadata] = {}
        self.scores: dict[str, int] = {}
        self.history: OrderedDict[str, AgentHistory] = OrderedDict()
        self.started: dict[str, float] = {}
        # One charge per occupied physical block, including shared prefix hits.
        # Values name the reservation owner, or None for shared capacity.
        self.charges: dict[int, str | None] = {}
        self._used: Counter[str | None] = Counter()
        self.plan = CapacityPlan()
        self.reserve_ratio = settings.reserve_ratio_min
        self.step = 0
        self.waiting_critical: set[str] = set()
        self.shared_available = 0
        self.reserved_available: dict[str, int] = {}
        self._accounting_active = False
        self._executed_ranges: dict[str, list[tuple[int, int]]] = {}
        self._admitted: dict[str, int] = {}
        self._growth_commitments: dict[str, int] = {}
        self._retry_capacity: dict[str, tuple[int, int]] = {}
        self._queued_at: dict[str, float] = {}
        self._completion_epoch = 0
        self.reservation_deferred: set[str] = set()
        self._cache_preferred: set[str] = set()

    def associate(self, request_id: str, metadata: TokenCakeMetadata) -> None:
        self.metadata[request_id] = metadata
        self._queued_at[request_id] = time.monotonic()

    def _history(self, key: str) -> AgentHistory:
        history = self.history.setdefault(key, AgentHistory())
        self.history.move_to_end(key)
        while len(self.history) > _HISTORY_LIMIT:
            self.history.popitem(last=False)
        return history

    def begin_step(self, waiting: Iterable[Request], running: list[Request]) -> bool:
        self.reservation_deferred.clear()
        self._cache_preferred.clear()
        if not self.metadata:
            self.charges.clear()
            self._used.clear()
            self.scores.clear()
            self._admitted.clear()
            self._growth_commitments.clear()
            self._retry_capacity.clear()
            self._accounting_active = False
            return False
        now = time.time()
        waiting = list(waiting)
        requests = [*waiting, *running]
        for request in running:
            self._admitted.setdefault(request.request_id, 0)
        self._growth_commitments = {
            r.request_id: self._remaining_growth(r)
            for r in requests
            if r.request_id in self._admitted
        }
        self.scores = {
            r.request_id: request_score(
                self.metadata[r.request_id], r.arrival_time, now
            )
            for r in requests
            if r.request_id in self.metadata
        }
        self.step += 1
        # Adopt allocations made before annotated work joined the engine.
        if not self._accounting_active:
            for request in requests:
                for block_id in self._block_ids(request):
                    self.charges.setdefault(block_id, None)
            self._accounting_active = True
        waiting_counts: Counter[str] = Counter()
        waiting_time: defaultdict[str, float] = defaultdict(float)
        for request in waiting:
            metadata = self.metadata.get(request.request_id)
            if metadata is not None and metadata.agent_type:
                waiting_counts[metadata.agent_type] += 1
                waiting_time[metadata.agent_type] += max(
                    0.0, now - request.arrival_time
                )
        if not self.plan.scores or self.step % _ADJUSTMENT_WINDOW == 0:
            importance = {key: value.importance for key, value in self.history.items()}
            used: Counter[str] = Counter()
            for request in requests:
                metadata = self.metadata.get(request.request_id)
                if metadata is not None and metadata.agent_type:
                    key = metadata.agent_type
                    importance[key] = max(importance.get(key, 0), metadata.importance)
                    used[key] += len(self._block_ids(request))
            scores = {
                key: self.history.get(key, AgentHistory()).score(
                    value,
                    waiting_counts[key],
                    waiting_time[key] / max(1, waiting_counts[key]),
                )
                for key, value in importance.items()
            }
            if self.manager.usage >= self.settings.gpu_usage_high:
                self.reserve_ratio += self.settings.reserve_adjustment_step
            elif self.manager.usage <= self.settings.gpu_usage_low:
                self.reserve_ratio -= self.settings.reserve_adjustment_step
            self.reserve_ratio = min(
                self.settings.reserve_ratio_max,
                max(self.settings.reserve_ratio_min, self.reserve_ratio),
            )
            self.plan = partition_capacity(
                scores,
                importance,
                used,
                len(self.manager.block_pool.blocks) - 1,
                self.reserve_ratio,
                self.settings.critical_ratio,
            )
        self.waiting_critical = self.plan.critical.intersection(waiting_counts)
        self.release()
        return True

    def _block_ids(self, request: Request) -> set[int]:
        return {
            block.block_id
            for group in self.manager.get_blocks(request.request_id).blocks
            for block in group
            if not block.is_null
        }

    def admission_tokens(self, request: Request) -> int:
        full_tokens = request.num_tokens
        if self.settings.reserve_generation_tokens:
            full_tokens = max(
                full_tokens, request.num_prompt_tokens + request.max_tokens
            )
        return min(full_tokens, self.manager.max_model_len)

    @property
    def uncommitted_blocks(self) -> int:
        return self.manager.block_pool.get_num_free_blocks() - sum(
            self._growth_commitments.values()
        )

    def _remaining_growth(self, request: Request) -> int:
        full_tokens = self.admission_tokens(request)
        if request.num_computed_tokens >= full_tokens:
            return 0
        return self.manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=min(
                full_tokens + self.num_lookahead_tokens, self.manager.max_model_len
            ),
            new_computed_blocks=self.manager.empty_kv_cache_blocks.blocks,
            num_encoder_tokens=self._admitted.get(request.request_id, 0),
            total_computed_tokens=request.num_computed_tokens,
            num_tokens_main_model=full_tokens,
            apply_admission_cap=True,
        )

    def release(self) -> None:
        if not self._accounting_active:
            return
        pool = self.manager.block_pool
        self.charges = {
            block_id: owner
            for block_id, owner in self.charges.items()
            if pool.blocks[block_id].ref_cnt > 0
        }
        self._used = Counter(self.charges.values())
        self._update_available()

    def _update_available(self) -> None:
        used = self._used
        self.shared_available = max(0, self.plan.shared - used[None])
        self.reserved_available = {
            key: max(0, count - used[key]) for key, count in self.plan.reserved.items()
        }
        excess = max(
            0,
            self.shared_available
            + sum(self.reserved_available.values())
            - max(0, self.uncommitted_blocks),
        )
        reduction = min(excess, self.shared_available)
        self.shared_available -= reduction
        excess -= reduction
        for key in sorted(self.reserved_available, key=lambda k: self.plan.scores[k]):
            reduction = min(excess, self.reserved_available[key])
            self.reserved_available[key] -= reduction
            excess -= reduction

    def prepare_running(self, request: Request) -> None:
        free = self.manager.block_pool.get_num_free_blocks()
        self.manager.remove_skipped_blocks(
            request.request_id, request.num_computed_tokens
        )
        if free != self.manager.block_pool.get_num_free_blocks():
            self.release()

    def prefill_budget(self, running: list[Request], native_budget: int) -> int:
        limit = self.settings.decode_prefill_token_budget
        if limit and any(r.num_computed_tokens >= r.num_prompt_tokens for r in running):
            return min(native_budget, limit)
        return native_budget

    def cap_prefill(self, tokens: int, budget: int) -> int:
        if tokens > budget:
            self.metrics.count(Metric.PREFILL_CAPPED)
            return budget
        return tokens

    def _consumption(
        self, request: Request, demand: int, *, borrow_reserved: bool = False
    ) -> list[tuple[str | None, int]] | None:
        shared = min(demand, self.shared_available)
        usage: list[tuple[str | None, int]] = [(None, shared)]
        remaining = demand - shared
        if remaining == 0:
            return usage
        metadata = self.metadata.get(request.request_id)
        if metadata is None:
            return None
        key = metadata.agent_type
        if key in self.plan.critical:
            own = min(remaining, self.reserved_available.get(key, 0))
            usage.append((key, own))
            remaining -= own
            if remaining == 0:
                return usage
        if not borrow_reserved and (
            self.waiting_critical - {key}
            or (key not in self.plan.critical and self.waiting_critical)
        ):
            return None
        donors = sorted(
            (k for k in self.reserved_available if k != key),
            key=lambda k: (self.reserved_available[k], self.plan.scores[k]),
            reverse=True,
        )
        for owner in donors:
            borrowed = min(remaining, self.reserved_available[owner])
            usage.append((owner, borrowed))
            remaining -= borrowed
        return usage if remaining == 0 else None

    def can_allocate(
        self,
        request: Request,
        num_new_tokens: int,
        *,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        num_encoder_tokens: int = 0,
        borrow_reserved: bool = False,
    ) -> bool:
        # An asynchronous cache load is already admitted and must be allowed
        # to finish admission using the capacity committed before the load.
        if request.request_id in self._admitted:
            return True
        computed = (
            request.num_computed_tokens
            + num_new_computed_tokens
            + num_external_computed_tokens
        )
        main_tokens = min(computed, self.manager.max_model_len) + num_new_tokens
        free_before = self.manager.block_pool.get_num_free_blocks()
        self.manager.remove_skipped_blocks(request.request_id, computed)
        if free_before != self.manager.block_pool.get_num_free_blocks():
            self.release()
        demand = self.manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=min(
                main_tokens + num_lookahead_tokens, self.manager.max_model_len
            ),
            new_computed_blocks=(
                new_computed_blocks or self.manager.empty_kv_cache_blocks
            ).blocks,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=computed,
            num_tokens_main_model=main_tokens,
        )
        full_tokens = self.admission_tokens(request)
        full_demand = self.manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=min(
                full_tokens + self.num_lookahead_tokens, self.manager.max_model_len
            ),
            new_computed_blocks=(
                new_computed_blocks or self.manager.empty_kv_cache_blocks
            ).blocks,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=computed,
            num_tokens_main_model=full_tokens,
            apply_admission_cap=True,
        )
        demand = max(demand, full_demand)
        available = self.uncommitted_blocks
        if demand > available:
            self.metrics.count(
                Metric.GENERATION_CAPACITY_DENIED
                if self.settings.reserve_generation_tokens
                else Metric.PREFILL_CAPACITY_DENIED
            )
            return False
        retry = self._retry_capacity.get(request.request_id)
        if (
            retry is not None
            and self._admitted
            and retry[1] == self._completion_epoch
            and available <= retry[0]
        ):
            self.metrics.count(Metric.RESUME_DEFERRED)
            return False
        allowed = (
            self._consumption(request, demand, borrow_reserved=borrow_reserved)
            is not None
        )
        if not allowed:
            self.metrics.count(Metric.RESERVATION_DENIED)
            if request.request_id in self.metadata:
                self.reservation_deferred.add(request.request_id)
        return allowed

    def commit(
        self,
        request: Request,
        *,
        running: bool = False,
        num_encoder_tokens: int = 0,
        new_blocks: KVCacheBlocks | None = None,
        borrow_reserved: bool = False,
    ) -> None:
        block_ids = sorted(
            (
                {
                    b.block_id
                    for group in new_blocks.blocks
                    for b in group
                    if not b.is_null
                }
                if running and new_blocks is not None
                else self._block_ids(request)
            )
            - self.charges.keys()
        )
        usage = self._consumption(
            request, len(block_ids), borrow_reserved=borrow_reserved
        )
        if usage is None and (running or request.request_id in self._admitted):
            # Accepted work may grow past a revised partition. Charge that debt
            # to shared capacity; later admissions pay it down as blocks free.
            usage = [(None, len(block_ids))]
        assert usage is not None, "Native allocation exceeded admitted capacity"
        offset = 0
        for owner, count in usage:
            for block_id in block_ids[offset : offset + count]:
                self.charges[block_id] = owner
            self._used[owner] += count
            offset += count
            if owner is None:
                self.shared_available = max(0, self.shared_available - count)
            else:
                self.reserved_available[owner] -= count
        self._admitted[request.request_id] = max(
            self._admitted.get(request.request_id, 0), num_encoder_tokens
        )
        self._growth_commitments[request.request_id] = self._remaining_growth(request)
        self._retry_capacity.pop(request.request_id, None)
        self.reservation_deferred.discard(request.request_id)
        if borrow_reserved and not running:
            self.metrics.count(Metric.WORK_CONSERVING_ADMITTED)
        if not running and request.request_id in self._cache_preferred:
            self.metrics.count(Metric.CACHE_AFFINITY_ADMITTED)
            self._cache_preferred.discard(request.request_id)
        self._update_available()
        metadata = self.metadata.get(request.request_id)
        if metadata is None:
            return
        queued_at = self._queued_at.pop(request.request_id, None)
        if queued_at is not None:
            self.metrics.admission_wait(
                max(0, time.monotonic() - queued_at),
                critical=metadata.critical_path
                or metadata.agent_type in self.plan.critical,
            )
        if metadata.agent_type:
            history = self._history(metadata.agent_type)
            history.deferrals = max(0, history.deferrals - 1)
            history.importance = metadata.importance
            if request.request_id not in self.started:
                history.samples += 1
                history.input_tokens += request.num_prompt_tokens
                history.depth += metadata.depth
                history.out_degree += metadata.out_degree
                history.in_degree += metadata.in_degree
                history.elapsed += metadata.application_elapsed_s
                history.remaining += metadata.remaining_depth
        self.started.setdefault(request.request_id, time.monotonic())

    def defer(self, request: Request) -> None:
        self.metrics.count(Metric.DEFERRED)
        metadata = self.metadata.get(request.request_id)
        if metadata is not None and metadata.agent_type:
            self._history(metadata.agent_type).deferrals += 1

    def preempt(self, request: Request, *, physical: bool = False) -> None:
        self._admitted.pop(request.request_id, None)
        self._growth_commitments.pop(request.request_id, None)
        self.release()
        self.metrics.count(Metric.PREEMPTED)
        if physical:
            self.metrics.count(Metric.PHYSICAL_PREEMPTED)
            self._retry_capacity[request.request_id] = (
                self.uncommitted_blocks,
                self._completion_epoch,
            )
        self._queued_at[request.request_id] = time.monotonic()
        metadata = self.metadata.get(request.request_id)
        if metadata is not None and metadata.agent_type:
            self._history(metadata.agent_type).preemptions += 1

    def finish(self, request: Request) -> None:
        was_admitted = request.request_id in self._admitted
        self._admitted.pop(request.request_id, None)
        self._growth_commitments.pop(request.request_id, None)
        self._retry_capacity.pop(request.request_id, None)
        self._queued_at.pop(request.request_id, None)
        if was_admitted:
            self._completion_epoch += 1
        self._executed_ranges.pop(request.request_id, None)
        metadata = self.metadata.pop(request.request_id, None)
        started = self.started.pop(request.request_id, None)
        self.scores.pop(request.request_id, None)
        if metadata is not None and metadata.agent_type and started is not None:
            history = self._history(metadata.agent_type)
            history.importance = metadata.importance
            history.output_tokens += request.num_output_tokens
            history.completed += 1
            history.duration += max(0.0, time.monotonic() - started)

    def cache_hits(
        self,
        request: Request,
        gpu: int,
        cpu: int,
        blocks: KVCacheBlocks | None = None,
    ) -> None:
        self.metrics.count(Metric.GPU_HIT_TOKENS, gpu)
        self.metrics.count(Metric.CPU_HIT_TOKENS, cpu)
        if blocks is not None:
            matched = {
                block.block_id: block
                for group in blocks.blocks
                for block in group
                if not block.is_null
            }
            shared = sum(block.ref_cnt > 1 for block in matched.values())
            self.metrics.count(Metric.GPU_SHARED_HIT_BLOCKS, shared)
            self.metrics.count(Metric.GPU_EXCLUSIVE_HIT_BLOCKS, len(matched) - shared)
        if request.num_preemptions:
            self.metrics.count(Metric.RESUME_GPU_HIT_TOKENS, gpu)
            self.metrics.count(Metric.RESUME_CPU_HIT_TOKENS, cpu)

    def executed(
        self,
        output: "SchedulerOutput",
        result: "ModelRunnerOutput",
        failed_loads: set[str] | None,
    ) -> None:
        # Observe returned model work, after scheduler rollback, using the
        # positions saved in this output rather than mutable async progress.
        cached = output.scheduled_cached_reqs
        starts = dict(zip(cached.req_ids, cached.num_computed_tokens))
        starts.update(
            (r.req_id, r.num_computed_tokens) for r in output.scheduled_new_reqs
        )
        for request_id, count in output.num_scheduled_tokens.items():
            if (
                request_id not in self.metadata
                or (failed_loads and request_id in failed_loads)
                or request_id not in result.req_id_to_index
            ):
                continue
            start = starts[request_id]
            end = start + count
            ranges = self._executed_ranges.setdefault(request_id, [])
            repeated = sum(max(0, min(end, b) - max(start, a)) for a, b in ranges)
            self.metrics.count(Metric.EXECUTED_TOKENS, count)
            self.metrics.count(Metric.RECOMPUTED_TOKENS, repeated)
            merged: list[tuple[int, int]] = []
            for a, b in sorted([*ranges, (start, end)]):
                if merged and a <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], b))
                else:
                    merged.append((a, b))
            self._executed_ranges[request_id] = merged

    def order_key(self, request: Request) -> tuple[int, int, float, str]:
        return (
            -self.scores[request.request_id],
            request.priority,
            request.arrival_time,
            request.request_id,
        )

    def _active_prefix_affinity(self, request: Request) -> tuple[float, int]:
        if (
            request.num_computed_tokens
            or request.skip_reading_prefix_cache
            or request.has_encoder_inputs
        ):
            return 0.0, 0
        blocks, computed = self.manager.coordinator.find_longest_cache_hit(
            request.block_hashes, request.num_tokens - 1
        )
        shared = len(
            {
                block.block_id
                for group in blocks
                for block in group
                if not block.is_null and block.ref_cnt > 0
            }
        )
        if not shared:
            return 0.0, 0
        full_tokens = self.admission_tokens(request)
        demand = self.manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=min(
                full_tokens + self.num_lookahead_tokens, self.manager.max_model_len
            ),
            new_computed_blocks=blocks,
            num_encoder_tokens=0,
            total_computed_tokens=computed,
            num_tokens_main_model=full_tokens,
            apply_admission_cap=True,
        )
        return shared / max(1, shared + demand), shared

    def order_candidates(
        self, candidates: list[tuple[Request, RequestQueue]]
    ) -> list[tuple[Request, RequestQueue]]:
        ordered = sorted(candidates, key=lambda item: self.order_key(item[0]))
        band = self.settings.cache_affinity_score_band
        capacity = len(self.manager.block_pool.blocks) - 1
        if (
            not band
            or not self.manager.enable_caching
            or self.uncommitted_blocks > capacity * (1 - self.settings.gpu_usage_high)
        ):
            return ordered
        result: list[tuple[Request, RequestQueue]] = []
        start = 0
        while start < len(ordered):
            highest = self.scores[ordered[start][0].request_id]
            stop = start + 1
            while (
                stop < len(ordered)
                and self.scores[ordered[stop][0].request_id] >= highest - band
            ):
                stop += 1
            # Score bands bound each promotion independently of memory units.
            group = ordered[start:stop]
            preferred = sorted(
                group,
                key=lambda item: self._active_prefix_affinity(item[0]),
                reverse=True,
            )
            positions = {item[0].request_id: i for i, item in enumerate(group)}
            self._cache_preferred.update(
                item[0].request_id
                for i, item in enumerate(preferred)
                if positions[item[0].request_id] > i
            )
            result.extend(preferred)
            start = stop
        return result

    def annotated_prefix(
        self, waiting: RequestQueue, skipped: RequestQueue, policy: SchedulingPolicy
    ) -> Iterator[tuple[Request, RequestQueue]]:
        skipped_items = ((r, skipped) for r in skipped)
        waiting_items = ((r, waiting) for r in waiting)
        merged = (
            chain(skipped_items, waiting_items)
            if policy == SchedulingPolicy.FCFS
            else heapq.merge(skipped_items, waiting_items, key=lambda item: item[0])
        )
        for request, queue in merged:
            if request.request_id not in self.metadata:
                break
            yield request, queue

    def victim(
        self,
        request: Request,
        running: list[Request],
        num_new_tokens: int = 1,
        num_lookahead_tokens: int = 0,
    ) -> Request | None:
        # Keep native victim selection for ordinary requesting work. Agent
        # scores are comparable only among annotated requests.
        if request.request_id not in self.metadata:
            return None
        candidates = [r for r in running if r.request_id in self.scores]
        lowest = min(self.scores[r.request_id] for r in candidates)
        candidates = [
            r
            for r in candidates
            if self.scores[r.request_id] <= lowest + _PREEMPTION_SCORE_BAND
        ]
        main_tokens = request.num_computed_tokens + num_new_tokens
        needed = (
            self.manager.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=min(
                    main_tokens + num_lookahead_tokens, self.manager.max_model_len
                ),
                new_computed_blocks=self.manager.empty_kv_cache_blocks.blocks,
                num_encoder_tokens=0,
                total_computed_tokens=request.num_computed_tokens,
                num_tokens_main_model=main_tokens,
            )
            - self.manager.block_pool.get_num_free_blocks()
        )
        pool = self.manager.block_pool
        released = {
            r.request_id: sum(pool.blocks[b].ref_cnt == 1 for b in self._block_ids(r))
            for r in candidates
        }
        sufficient = [
            r
            for r in candidates
            if r is not request and released[r.request_id] >= needed
        ]
        if sufficient:
            candidates = sufficient
        else:
            positive = [r for r in candidates if released[r.request_id] > 0]
            if positive:
                candidates = positive

        def cost(r: Request) -> tuple:
            computed = r.num_computed_tokens
            near_finish = r.num_output_tokens >= 0.9 * r.max_tokens
            return (
                near_finish,
                computed / max(1, released[r.request_id]),
                computed,
                -released[r.request_id],
                (self.scores[r.request_id], -r.priority, -r.arrival_time, r.request_id),
            )

        return min(candidates, key=cost)
