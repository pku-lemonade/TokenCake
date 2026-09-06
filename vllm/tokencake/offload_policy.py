# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Snapshot preservation policy; no cache mutation or transfer submission."""

import heapq
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING

from vllm.tokencake.config import OffloadConfig
from vllm.tokencake.metrics import Metric
from vllm.tokencake.protocol import TokenCakeMetadata
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler


@dataclass(frozen=True)
class Pressure:
    usage: float
    free: int
    waiting_count: int
    waiting_demand: int
    critical_demand: int
    shared_available: int
    fit_demand: int

    @property
    def signature(self) -> tuple[int, ...]:
        return (
            self.free,
            self.waiting_count,
            self.waiting_demand,
            self.critical_demand,
            self.shared_available,
            self.fit_demand,
        )


def waiting_pressure(
    scheduler: "Scheduler", window: int, temporal_selection: str
) -> Pressure:
    manager = scheduler.kv_cache_manager
    controller = scheduler._tokencake_scheduling
    if controller is not None and not controller.metadata:
        controller = None
    free = manager.block_pool.get_num_free_blocks()
    uncommitted = controller.uncommitted_blocks if controller is not None else free
    shared = controller.shared_available if controller is not None else free
    critical = controller.plan.critical if controller is not None else set()
    if scheduler.policy == SchedulingPolicy.FCFS:
        waiting = list(chain(scheduler.skipped_waiting, scheduler.waiting))
    else:
        waiting = list(heapq.merge(scheduler.skipped_waiting, scheduler.waiting))
    waiting_demand = 0
    critical_demand = 0
    candidates = []
    token_budget = scheduler.max_num_scheduled_tokens
    if (
        controller is not None
        and scheduler.scheduler_config.enable_chunked_prefill
        and not scheduler.need_mamba_block_aligned_split
    ):
        token_budget = controller.prefill_budget(scheduler.running, token_budget)
    group_block_sizes = [m.block_size for m in manager.coordinator.single_type_managers]
    loras = {r.lora_request.lora_int_id for r in scheduler.running if r.lora_request}
    for request in waiting:
        remaining = max(0, request.num_tokens - request.num_computed_tokens)
        total_demand = sum((remaining + size - 1) // size for size in group_block_sizes)
        waiting_demand += total_demand
        metadata = controller.metadata.get(request.request_id) if controller else None
        if metadata is not None and metadata.agent_type in critical:
            critical_demand += total_demand
        if request.status not in (RequestStatus.WAITING, RequestStatus.PREEMPTED):
            continue
        if (
            scheduler.lora_config
            and request.lora_request
            and len(loras) == scheduler.lora_config.max_loras
            and request.lora_request.lora_int_id not in loras
        ):
            continue
        computed = request.num_computed_tokens
        blocks = manager.empty_kv_cache_blocks.blocks
        if (
            computed == 0
            and manager.enable_caching
            and not request.skip_reading_prefix_cache
        ):
            blocks, computed = manager.coordinator.find_longest_cache_hit(
                request.block_hashes, request.num_tokens - 1
            )
        tokens = max(0, request.num_tokens - computed)
        threshold = scheduler.scheduler_config.long_prefill_token_threshold
        if threshold > 0:
            tokens = min(tokens, threshold)
        limited_prefill = (
            controller is not None
            and request.request_id in controller.metadata
            and not request.has_encoder_inputs
        )
        budget = token_budget if limited_prefill else scheduler.max_num_scheduled_tokens
        if not scheduler.scheduler_config.enable_chunked_prefill and tokens > budget:
            continue
        tokens = min(tokens, budget, scheduler.max_model_len - computed)
        if tokens <= 0:
            continue
        encoder_tokens = 0
        if request.has_encoder_inputs:
            cache = scheduler.encoder_cache_manager
            required = {
                feature.identifier: feature.mm_position.get_num_embeds()
                for feature in request.mm_features
                if feature.mm_position.offset < computed + tokens
                and feature.mm_position.offset + feature.mm_position.length > computed
            }
            missing = sum(n for key, n in required.items() if key not in cache.cached)
            protected = sum(cache.freeable.get(key, 0) for key in required)
            if (
                missing > scheduler.max_num_encoder_input_tokens
                or missing + protected > cache.num_freeable_slots
            ):
                continue
            if scheduler.scheduler_config.disable_chunked_mm_input and any(
                computed < feature.mm_position.offset
                and feature.mm_position.offset
                < computed + tokens
                < feature.mm_position.offset + feature.mm_position.length
                for feature in request.mm_features
            ):
                continue
            if scheduler.is_encoder_decoder:
                encoder_tokens = sum(required.values())
        lookahead = scheduler.num_lookahead_tokens if request.num_computed_tokens else 0
        demand = manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=min(computed + tokens + lookahead, scheduler.max_model_len),
            new_computed_blocks=blocks,
            num_encoder_tokens=encoder_tokens,
            total_computed_tokens=computed,
            num_tokens_main_model=computed + tokens,
        )
        admission = demand
        if scheduler.scheduler_reserve_full_isl or controller is not None:
            full_tokens = (
                controller.admission_tokens(request)
                if controller is not None
                else min(request.num_tokens, scheduler.max_model_len)
            )
            full_lookahead = (
                controller.num_lookahead_tokens if controller is not None else 0
            )
            full_demand = manager.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=min(full_tokens + full_lookahead, scheduler.max_model_len),
                new_computed_blocks=blocks,
                num_encoder_tokens=encoder_tokens,
                total_computed_tokens=computed,
                num_tokens_main_model=full_tokens,
                apply_admission_cap=True,
            )
            admission = max(admission, full_demand)
        if demand <= 0 or demand > free or admission > uncommitted:
            continue
        borrow_reserved = False
        if (
            controller is not None
            and controller._consumption(request, admission) is None
        ):
            if (
                controller._consumption(request, admission, borrow_reserved=True)
                is None
            ):
                continue
            borrow_reserved = True
        score = (
            controller.scores.get(request.request_id, 0)
            if controller is not None
            else -request.priority
        )
        candidates.append(
            (
                request,
                demand,
                tokens,
                score,
                admission,
                borrow_reserved,
                limited_prefill,
            )
        )
    if temporal_selection == "priority_first":
        candidates.sort(key=lambda c: (c[3], c[1], -c[0].arrival_time), reverse=True)
    elif temporal_selection == "best_fit":
        candidates.sort(
            key=lambda c: (-abs(window - c[1]), c[3], -c[0].arrival_time), reverse=True
        )
    candidates.sort(key=lambda c: c[5])
    slots = scheduler.max_num_running_reqs - len(scheduler.running)
    remaining_blocks = free
    remaining_commitment = uncommitted
    remaining_tokens = scheduler.max_num_scheduled_tokens
    remaining_prefill = token_budget
    fit = 0
    if scheduler.pause_state == PauseState.UNPAUSED:
        for _, demand, tokens, _, admission, _, limited_prefill in candidates:
            if slots <= 0:
                break
            if (
                demand > remaining_blocks
                or admission > remaining_commitment
                or tokens > remaining_tokens
                or (limited_prefill and tokens > 1 and tokens > remaining_prefill)
            ):
                continue
            slots -= 1
            remaining_blocks -= demand
            remaining_commitment -= admission
            remaining_tokens -= tokens
            if limited_prefill and tokens > 1:
                remaining_prefill -= tokens
            fit += demand
    return Pressure(
        manager.usage,
        free,
        len(waiting),
        waiting_demand,
        critical_demand,
        shared,
        min(window, fit),
    )


@dataclass(frozen=True)
class Benefit:
    reason: Metric
    score: float


def evaluate_benefit(
    settings: OffloadConfig,
    metadata: TokenCakeMetadata,
    pressure: Pressure,
    *,
    blocks: int,
    cpu_available: int,
    cpu_needed: int,
    duration: float,
    transfer_time: float,
    agent_score: float = 0.0,
    churn_penalty: float = 0.0,
) -> Benefit:
    """Latest-source utility for a completed request's detached metadata."""
    importance = min(10.0, max(metadata.importance, min(10.0, agent_score / 2)))
    near_completion = metadata.near_completion or (
        metadata.application_max_depth > 0 and metadata.remaining_depth <= 1
    )
    margin = duration - transfer_time
    pressure_score = max(0.0, (pressure.usage - settings.min_gpu_usage) * 6) + min(
        2.0, pressure.waiting_demand / max(1, blocks + 1)
    )
    fit_score = min(2.0, pressure.fit_demand / max(1, blocks))
    stall_score = min(6.0, margin / max(0.25, transfer_time))
    upload_safety = 1.0 if duration >= 2 * transfer_time else -1.0
    cpu_score = 1.0 if cpu_available >= cpu_needed else -4.0
    score = (
        stall_score
        + pressure_score
        + fit_score
        + upload_safety
        + cpu_score
        - 0.35 * importance
        - (2.0 if near_completion else 0.0)
        - churn_penalty
    )
    emergency = (
        pressure.usage >= settings.high_pressure_gpu_usage
        and margin >= max(4 * transfer_time, 5.0)
        and pressure.waiting_count > 0
    )
    low_pressure = pressure.usage < settings.min_gpu_usage
    no_queue_pressure = (
        pressure.waiting_demand <= pressure.free
        and pressure.critical_demand <= pressure.shared_available
    )
    reason = Metric.SELECTED
    if not metadata.offload_eligible or not metadata.reusable_prefix:
        reason = Metric.NOT_ELIGIBLE
    elif blocks <= 0:
        reason = Metric.EMPTY
    elif pressure.fit_demand <= 0:
        reason = Metric.NO_WAITING_DEMAND
    elif cpu_available < cpu_needed:
        reason = Metric.CPU_CAPACITY
    elif duration <= transfer_time or (
        margin < max(0.05, 2 * transfer_time) and not emergency
    ):
        reason = Metric.UNPROFITABLE
    elif low_pressure and (
        no_queue_pressure or pressure.fit_demand < blocks or pressure.waiting_count == 0
    ):
        reason = Metric.LOW_PRESSURE
    elif (
        importance >= 8.0
        and not emergency
        and (
            near_completion or margin < max(0.5, 8 * transfer_time) or upload_safety < 0
        )
        or score < settings.score_threshold
        and not emergency
    ):
        reason = Metric.UNPROFITABLE
    return Benefit(reason, score)
