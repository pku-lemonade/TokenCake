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

from vllm.tokencake.config import SchedulingConfig
from vllm.tokencake.metrics import Metric, TokenCakeMetrics
from vllm.tokencake.protocol import TokenCakeMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.request_queue import RequestQueue, SchedulingPolicy
from vllm.v1.request import Request

_ADJUSTMENT_WINDOW = 500
_HISTORY_LIMIT = 4096
_PREEMPTION_MARGIN = 3000


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
    ) -> None:
        self.settings = settings
        self.manager = manager
        self.metrics = metrics
        self.metadata: dict[str, TokenCakeMetadata] = {}
        self.scores: dict[str, int] = {}
        self.history: OrderedDict[str, AgentHistory] = OrderedDict()
        self.started: dict[str, float] = {}
        # One charge per occupied physical block, including shared prefix hits.
        # Values name the reservation owner, or None for shared capacity.
        self.charges: dict[int, str | None] = {}
        self.plan = CapacityPlan()
        self.reserve_ratio = settings.reserve_ratio_min
        self.step = 0
        self.waiting_critical: set[str] = set()
        self.shared_available = 0
        self.reserved_available: dict[str, int] = {}
        self._accounting_active = False

    def associate(self, request_id: str, metadata: TokenCakeMetadata) -> None:
        self.metadata[request_id] = metadata

    def _history(self, key: str) -> AgentHistory:
        history = self.history.setdefault(key, AgentHistory())
        self.history.move_to_end(key)
        while len(self.history) > _HISTORY_LIMIT:
            self.history.popitem(last=False)
        return history

    def begin_step(self, waiting: Iterable[Request], running: list[Request]) -> bool:
        if not self.metadata:
            self.charges.clear()
            self.scores.clear()
            self._accounting_active = False
            return False
        now = time.time()
        waiting = list(waiting)
        requests = [*waiting, *running]
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

    def release(self) -> None:
        if not self._accounting_active:
            return
        pool = self.manager.block_pool
        self.charges = {
            block_id: owner
            for block_id, owner in self.charges.items()
            if pool.blocks[block_id].ref_cnt > 0
        }
        used = Counter(self.charges.values())
        self.shared_available = max(0, self.plan.shared - used[None])
        self.reserved_available = {
            key: max(0, count - used[key]) for key, count in self.plan.reserved.items()
        }
        excess = max(
            0,
            self.shared_available
            + sum(self.reserved_available.values())
            - pool.get_num_free_blocks(),
        )
        reduction = min(excess, self.shared_available)
        self.shared_available -= reduction
        excess -= reduction
        for key in sorted(self.reserved_available, key=lambda k: self.plan.scores[k]):
            reduction = min(excess, self.reserved_available[key])
            self.reserved_available[key] -= reduction
            excess -= reduction

    def _consumption(
        self, request: Request, demand: int
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
        if self.waiting_critical - {key} or (
            key not in self.plan.critical and self.waiting_critical
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
    ) -> bool:
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
        # Physical exhaustion must reach native allocation/preemption. Protect
        # the capacity presently available; recheck after each native victim.
        available_demand = min(demand, self.manager.block_pool.get_num_free_blocks())
        return self._consumption(request, available_demand) is not None

    def commit(self, request: Request) -> None:
        block_ids = sorted(self._block_ids(request) - self.charges.keys())
        usage = self._consumption(request, len(block_ids))
        assert usage is not None, "Native allocation exceeded admitted capacity"
        offset = 0
        for owner, count in usage:
            for block_id in block_ids[offset : offset + count]:
                self.charges[block_id] = owner
            offset += count
            if owner is None:
                self.shared_available -= count
            else:
                self.reserved_available[owner] -= count
        metadata = self.metadata.get(request.request_id)
        if metadata is None:
            return
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

    def preempt(self, request: Request) -> None:
        self.release()
        self.metrics.count(Metric.PREEMPTED)
        metadata = self.metadata.get(request.request_id)
        if metadata is not None and metadata.agent_type:
            self._history(metadata.agent_type).preemptions += 1

    def finish(self, request: Request) -> None:
        metadata = self.metadata.pop(request.request_id, None)
        started = self.started.pop(request.request_id, None)
        self.scores.pop(request.request_id, None)
        if metadata is not None and metadata.agent_type and started is not None:
            history = self._history(metadata.agent_type)
            history.importance = metadata.importance
            history.output_tokens += request.num_output_tokens
            history.completed += 1
            history.duration += max(0.0, time.monotonic() - started)

    def order_key(self, request: Request) -> tuple[int, int, float, str]:
        return (
            -self.scores[request.request_id],
            request.priority,
            request.arrival_time,
            request.request_id,
        )

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

    def prefill_limit(
        self, request: Request, tokens: int, backlog: bool, running: int, maximum: int
    ) -> int:
        if (
            tokens > 256
            and request.request_id in self.metadata
            and (
                backlog or running >= max(2, maximum // 2) or self.manager.usage >= 0.8
            )
        ):
            self.metrics.count(Metric.PREFILL_CAPPED)
            return 256
        return tokens

    def victim(self, request: Request, running: list[Request]) -> Request | None:
        # Keep native victim selection for ordinary requesting work. Agent
        # scores are comparable only among annotated requests.
        if request.request_id not in self.metadata:
            return None
        candidates = [r for r in running if r.request_id in self.scores]
        victim = max(candidates, key=self.order_key)
        if (
            victim is not request
            and self.scores[request.request_id]
            < self.scores[victim.request_id] + _PREEMPTION_MARGIN
        ):
            return request
        return victim
