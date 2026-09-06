# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TokenCake's scheduler-side extension of native CPU offload."""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, replace
from math import lcm
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
    OffloadingOperationMetrics,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
    RequestOffloadState,
    TransferJobStatus,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.tokencake.lifecycle import (
    GenerationCause,
    Lifecycle,
    LifecycleRegistry,
    PrefixSnapshot,
    SnapshotGroup,
)
from vllm.tokencake.metrics import Metric
from vllm.tokencake.offload_policy import evaluate_benefit, waiting_pressure
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    OffloadingSpec,
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler


@dataclass(frozen=True)
class SnapshotBlock:
    group: int
    index: int
    key: OffloadKey
    block_ids: tuple[int, ...]
    block_index: int


@dataclass(frozen=True)
class DetachedStore:
    lifecycle_id: str
    context: ReqContext
    source_ids: list[int]


class TokenCakeOffloadingScheduler(OffloadingConnectorScheduler):
    manager: CPUOffloadingManager

    def __init__(self, spec: CPUOffloadingSpec) -> None:
        super().__init__(spec)
        if not isinstance(self.manager, CPUOffloadingManager):
            raise ValueError("TokenCake requires the concrete CPUOffloadingManager")
        settings = spec.vllm_config._tokencake_config
        assert settings is not None and settings.offload.enabled
        self.lifecycles = LifecycleRegistry(settings.offload, self.manager)
        self.block_pool: BlockPool | None = None
        self.settings = settings.offload
        self.temporal_selection = settings.temporal_selection
        self.num_cpu_blocks = spec.num_blocks
        self.bytes_per_block = spec.cpu_page_size_per_worker
        self._shareable_groups = {
            index
            for index, group in enumerate(spec.kv_cache_config.kv_cache_groups)
            if type(group.kv_cache_spec) is FullAttentionSpec
        }
        self.step = 0
        self._unpublished: dict[int, TransferJob] = {}
        self._detached: dict[int, DetachedStore] = {}
        self._recent_stores: deque[int] = deque(maxlen=96)
        self._bandwidth: dict[str, float] = {}

    def _defer_store(self, request: Request) -> bool:
        params = request.sampling_params
        return params is not None and "tokencake" in (params.extra_args or {})

    def _peek_ready(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        block = self.manager._policy.get(key)
        if block is None:
            return False
        return True if block.is_ready else None

    def estimate_recompute_tokens(self, request: Request) -> int:
        computed = max(0, request.num_computed_tokens)
        if not self._defer_store(request) or request.skip_reading_prefix_cache:
            return computed
        # Reuse native group alignment without touching live request state,
        # cache recency, reference counts, or store-frequency tracking.
        state = RequestOffloadState(config=self.config, req=request)
        state.update_offload_keys()
        recoverable = self._lookup(state, lookup=self._peek_ready)
        return max(0, computed - (recoverable or 0))

    def capture_snapshot(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> PrefixSnapshot:
        assert self.block_pool is not None
        groups = []
        factor = self.config.block_size_factor
        computed = min(request.num_computed_tokens, request.num_tokens)
        for config, ids in zip(self.config.kv_group_configs, block_ids):
            hashes = request.block_hashes[
                config.hash_block_size_factor - 1 :: config.hash_block_size_factor
            ][: computed // config.offloaded_block_size]
            blocks = tuple(
                tuple(ids[i * factor : (i + 1) * factor]) for i in range(len(hashes))
            )
            groups.append(
                SnapshotGroup(
                    keys=tuple(make_offload_key(h, config.group_idx) for h in hashes),
                    block_ids=blocks,
                    gpu_hashes=tuple(
                        tuple(self.block_pool.blocks[bid].block_hash for bid in group)
                        for group in blocks
                    ),
                )
            )
        return PrefixSnapshot(
            tuple(groups),
            ReqContext(request.request_id, deepcopy(request.kv_transfer_params)),
        )

    def generation_finished(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> None:
        params = request.sampling_params
        metadata = (
            (params.extra_args or {}).get("tokencake") if params is not None else None
        )
        if metadata is None:
            return
        identifier = metadata["lifecycle_id"]
        record = self.lifecycles.records.get(identifier)
        snapshot = None
        cause: GenerationCause = "completed"
        if request.status == RequestStatus.FINISHED_ABORTED:
            cause = "aborted"
        elif request.status in (
            RequestStatus.FINISHED_ERROR,
            RequestStatus.FINISHED_IGNORED,
        ):
            cause = "error"
        elif record is not None and record.terminal_at is None:
            snapshot = self.capture_snapshot(request, block_ids)
        self.lifecycles.generation_finished(identifier, cause, snapshot)

    def _snapshot_candidates(self, record: Lifecycle) -> list[SnapshotBlock]:
        assert record.snapshot is not None and self.block_pool is not None
        candidates = []
        factor = self.config.block_size_factor
        for group_index, (group, config) in enumerate(
            zip(record.snapshot.groups, self.config.kv_group_configs)
        ):
            window = config.sliding_window_size_in_blocks
            start = max(0, len(group.keys) - window) if window is not None else 0
            for index in range(start, len(group.keys)):
                key = group.keys[index]
                cpu = self.manager._policy.get(key)
                ids = group.block_ids[index]
                hashes = group.gpu_hashes[index]
                offset = 0
                if window is not None:
                    while offset < len(ids) and ids[offset] == 0:
                        offset += 1
                source_ids = ids[offset:]
                matches = (
                    bool(source_ids)
                    and len(ids) == factor
                    and all(
                        bid != 0
                        and expected is not None
                        and self.block_pool.blocks[bid].block_hash == expected
                        for bid, expected in zip(source_ids, hashes[offset:])
                    )
                )
                if cpu is None and not matches:
                    self.lifecycles.metrics.count(Metric.STALE)
                    self.lifecycles.metrics.count(Metric.PREFIX_GAP)
                    break
                if key in record.retained or key in record.pending_retention:
                    continue
                # Completed full-attention prefix blocks are immutable while
                # shared. Native fences also cover their eventual reuse.
                if (
                    cpu is None
                    and group_index not in self._shareable_groups
                    and any(
                        self.block_pool.blocks[bid].ref_cnt != 0 for bid in source_ids
                    )
                ):
                    continue
                candidates.append(
                    SnapshotBlock(
                        group_index, index, key, source_ids, index * factor + offset
                    )
                )
        candidates.sort(
            key=lambda c: (
                (c.index + 1)
                * self.config.kv_group_configs[c.group].offloaded_block_size,
                c.group,
            )
        )
        return candidates

    def _bound_candidates(
        self, candidates: list[SnapshotBlock], limit: int
    ) -> list[SnapshotBlock]:
        candidates = candidates[: limit // self.config.block_size_factor]
        configs = self.config.kv_group_configs
        if not candidates or (
            len(configs) == 1 and configs[0].sliding_window_size_in_blocks is None
        ):
            return candidates
        by_group = [
            {c.index for c in candidates if c.group == config.group_idx}
            for config in configs
        ]
        if not all(by_group):
            return []
        # Native HMA hits end at a common aligned token boundary. A truncated
        # group or sliding window must not turn a bounded store into dead data.
        alignment = lcm(*(c.offloaded_block_size for c in configs))
        end = min(
            (max(indices) + 1) * config.offloaded_block_size
            for config, indices in zip(configs, by_group)
        )
        end = end // alignment * alignment
        for config, indices in zip(configs, by_group):
            window = config.sliding_window_size_in_blocks
            if window is not None:
                stop = end // config.offloaded_block_size
                if not set(range(max(0, stop - window), stop)) <= indices:
                    return []
        return [
            c
            for c in candidates
            if (c.index + 1) * configs[c.group].offloaded_block_size <= end
        ]

    def estimate_transfer(self, blocks: int, direction: str) -> float:
        transfer = self.settings.transfer
        if direction == "d2h":
            bandwidth = transfer.d2h_bandwidth_gbps * 1e9
            base = transfer.d2h_base_time_s
        else:
            assert direction == "h2d"
            bandwidth = transfer.h2d_bandwidth_gbps * 1e9
            base = transfer.h2d_base_time_s
        bandwidth = self._bandwidth.get(direction, bandwidth)
        return (
            base
            + blocks * self.bytes_per_block / bandwidth
            + blocks * transfer.submission_time_per_run_s
        )

    def _cpu_available(self) -> int:
        protected = set()
        for status in self._jobs.values():
            protected.update(status.keys)
        for record in self.lifecycles.records.values():
            protected.update(record.retained)
        return max(0, self.num_cpu_blocks - len(protected))

    def _preservation_limit(
        self, candidates: list[SnapshotBlock], available: int, remaining: float
    ) -> int:
        new_count = 0
        limit = 0
        for count, candidate in enumerate(candidates, 1):
            new_count += self.manager._policy.get(candidate.key) is None
            transfer = self.estimate_transfer(
                new_count, "d2h"
            ) + self.estimate_transfer(count, "h2d")
            # Leave room for transfer uncertainty and a subsequent restore.
            if new_count > available or remaining - transfer < max(0.05, 2 * transfer):
                break
            limit = count * self.config.block_size_factor
        return limit

    def evaluate_pending(
        self, scheduler: "Scheduler", *, new_step: bool = False
    ) -> None:
        self.lifecycles.expire()
        if new_step:
            self.step += 1
        while self._recent_stores and self.step - self._recent_stores[0] > 4:
            self._recent_stores.popleft()
        pressure = None
        for record in list(self.lifecycles.records.values()):
            # This consumes the post-finish marker even when no policy work is useful.
            record.pending_evaluation = False
            if not record.owns_preservation:
                continue
            # Commit one capacity- and duration-bounded preservation batch.
            if record.retained or record.pending_retention:
                continue
            if (
                not record.metadata.offload_eligible
                or not record.metadata.reusable_prefix
            ):
                self.lifecycles.metrics.count(Metric.NOT_ELIGIBLE)
                continue
            if record.snapshot is None:
                self.lifecycles.metrics.count(Metric.EMPTY)
                continue
            if pressure is None:
                pressure = waiting_pressure(
                    scheduler,
                    self.settings.eviction_window_blocks,
                    self.temporal_selection,
                )
            available = self._cpu_available()
            signature = (*pressure.signature, available)
            if (
                record.backoff_signature == signature
                and self.step < record.next_evaluation_step
            ):
                self.lifecycles.metrics.count(Metric.BACKOFF)
                continue
            assert record.started_at is not None
            remaining = max(
                0.0,
                record.started_at + record.predicted_duration - self.lifecycles.clock(),
            )
            candidates = self._snapshot_candidates(record)
            limit = (
                min(pressure.fit_demand, self.settings.max_relief_blocks)
                if self.settings.max_relief_blocks
                else self._preservation_limit(candidates, available, remaining)
            )
            bounded = self._bound_candidates(candidates, limit)
            if candidates and limit > 0 and not bounded:
                self.lifecycles.metrics.count(Metric.PREFIX_GAP)
            candidates = bounded
            gpu_blocks = len(candidates) * self.config.block_size_factor
            new_count = sum(self.manager._policy.get(c.key) is None for c in candidates)
            transfer_time = self.estimate_transfer(
                new_count, "d2h"
            ) + self.estimate_transfer(len(candidates), "h2d")
            controller = scheduler._tokencake_scheduling
            agent_score = (
                controller.plan.scores.get(record.metadata.agent_type, 0.0)
                if controller is not None
                else 0.0
            )
            churn = 1.5 if self.step - record.last_store_step <= 4 else 0.0
            churn += min(4.0, max(0, len(self._recent_stores) - 64) / 8)
            decision = evaluate_benefit(
                self.settings,
                record.metadata,
                pressure,
                blocks=gpu_blocks,
                cpu_available=available,
                cpu_needed=new_count,
                duration=remaining,
                transfer_time=transfer_time,
                agent_score=agent_score,
                churn_penalty=churn,
            )
            reason = decision.reason
            if pressure.fit_demand == 0:
                reason = Metric.NO_WAITING_DEMAND
            if reason == Metric.SELECTED:
                self.lifecycles.set_h2d_estimate(
                    record,
                    self.estimate_transfer(
                        len(candidates) + len(record.retained), "h2d"
                    ),
                )
                if record.owns_preservation:
                    reason = self._register_store(record, candidates)
            self.lifecycles.metrics.count(reason)
            if reason == Metric.SELECTED:
                record.backoff_signature = None
            else:
                record.backoff_signature = signature
                record.next_evaluation_step = self.step + self.settings.backoff_steps
        self._observe_frontier()

    def _register_store(
        self, record: Lifecycle, candidates: list[SnapshotBlock]
    ) -> Metric:
        assert record.snapshot is not None
        context = record.snapshot.req_context
        ordered = sorted(candidates, key=lambda c: (c.group, c.index))
        keys = [c.key for c in ordered]
        prepared = self.manager.prepare_store(keys, context)
        if prepared is None:
            return Metric.CPU_CAPACITY
        self.lifecycles.retain_ready(record, keys)
        if not record.owns_preservation:
            self.manager.complete_store(prepared.keys_to_store, context, success=False)
            return Metric.UNPROFITABLE
        for key in keys:
            block = self.manager._policy.get(key)
            if block is not None and not block.is_ready:
                record.pending_retention.add(key)
        if not prepared.keys_to_store:
            return (
                Metric.SELECTED
                if record.retained or record.pending_retention
                else Metric.UNPROFITABLE
            )
        new_keys = set(prepared.keys_to_store)
        sources = []
        group_sizes = []
        block_indices = []
        for group in range(len(self.config.kv_group_configs)):
            selected = [c for c in ordered if c.group == group and c.key in new_keys]
            group_ids = [bid for candidate in selected for bid in candidate.block_ids]
            sources.extend(group_ids)
            group_sizes.append(len(group_ids))
            block_indices.append(selected[0].block_index if selected else 0)
        job_id = self._generate_job_id()
        self._jobs[job_id] = TransferJobStatus(
            req_id=context.req_id,
            pending_count=self.config.num_workers,
            keys=new_keys,
            is_store=True,
        )
        self._detached[job_id] = DetachedStore(
            record.metadata.lifecycle_id, context, sources
        )
        for bid in sources:
            self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)
        self._unpublished[job_id] = TransferJob(
            context.req_id,
            (
                GPULoadStoreSpec(sources, group_sizes, block_indices),
                prepared.store_spec,
            ),
        )
        record.last_store_step = self.step
        self._recent_stores.append(self.step)
        return Metric.SELECTED

    def _observe_frontier(self) -> None:
        assert self.block_pool is not None
        known = {
            block_hash
            for record in self.lifecycles.records.values()
            if record.owns_preservation and record.snapshot is not None
            for group in record.snapshot.groups
            for hashes in group.gpu_hashes
            for block_hash in hashes
            if block_hash is not None
        }
        if not known:
            return
        observed = self.block_pool.peek_free_block_frontier(
            self.settings.eviction_window_blocks
        )
        self.lifecycles.metrics.count(
            Metric.EXTERNAL,
            sum(
                block_hash is not None and block_hash not in known
                for _, block_hash in observed
            ),
        )

    @property
    def has_unpublished(self) -> bool:
        return bool(self._unpublished)

    @property
    def has_pending_work(self) -> bool:
        return bool(
            self._detached or self._unpublished or self._current_batch_jobs_to_flush
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> OffloadingConnectorMetadata:
        unpublished, self._unpublished = self._unpublished, {}
        if unpublished:
            assert scheduler_output.total_num_scheduled_tokens == 0
        meta = super().build_connector_meta(scheduler_output)
        assert isinstance(meta, OffloadingConnectorMetadata)
        meta.store_jobs.update(unpublished)
        assert meta.jobs_to_flush is not None
        # Newly published stores do not exist on the worker until the next step.
        meta.jobs_to_flush.difference_update(unpublished)
        if not scheduler_output.total_num_scheduled_tokens and not unpublished:
            meta.jobs_to_flush.update(self._detached)
        if meta.jobs_to_flush.intersection(self._detached):
            self.lifecycles.metrics.count(Metric.FENCE_WAIT)
        return meta

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        metadata = connector_output.kv_connector_worker_meta
        ordinary = {}
        completed_keys: set[OffloadKey] = set()
        if isinstance(metadata, OffloadingWorkerMetadata):
            for job_id, count in metadata.completed_jobs.items():
                detached = self._detached.get(job_id)
                if detached is None:
                    ordinary[job_id] = count
                    status = self._jobs.get(job_id)
                    if (
                        status is not None
                        and status.is_store
                        and status.pending_count == count
                    ):
                        completed_keys.update(status.keys)
                    continue
                assert count > 0
                status = self._jobs[job_id]
                status.pending_count -= count
                assert status.pending_count >= 0
                if status.pending_count:
                    continue
                try:
                    self.manager.complete_store(status.keys, detached.context)
                except Exception:
                    self.lifecycles.metrics.count(Metric.TRANSFER_FAILURE)
                    raise
                completed_keys.update(status.keys)
                self._remove_pending_job(job_id, detached.source_ids)
                del self._detached[job_id], self._jobs[job_id]
                self.lifecycles.metrics.count(Metric.SAVED, len(detached.source_ids))
            connector_output = replace(
                connector_output,
                kv_connector_worker_meta=OffloadingWorkerMetadata(ordinary),
            )
        super().update_connector_output(connector_output)
        self.lifecycles.expire()
        for record in list(self.lifecycles.records.values()):
            ready = record.pending_retention.intersection(completed_keys)
            if ready:
                self.lifecycles.retain_ready(record, ready)
                record.pending_retention.difference_update(ready)
        stats = connector_output.kv_connector_stats
        if isinstance(stats, OffloadingConnectorStats):
            for name, operations in stats.data.items():
                if name not in ("GPU_to_CPU", "CPU_to_GPU"):
                    continue
                direction = "d2h" if name == "GPU_to_CPU" else "h2d"
                for op in operations:
                    if isinstance(op, dict):
                        op = OffloadingOperationMetrics(**op)
                    if op.op_size <= 0 or op.op_time <= 0:
                        continue
                    sample = op.op_size / op.op_time
                    previous = self._bandwidth.get(direction, sample)
                    alpha = self.settings.ewma_alpha
                    self._bandwidth[direction] = alpha * sample + (1 - alpha) * previous

    def reset_cache(self) -> None:
        super().reset_cache()
        self._current_batch_jobs_to_flush.difference_update(self._unpublished)
        self._unpublished.clear()
        self._detached.clear()
        self._recent_stores.clear()
        self.lifecycles.external_reset_succeeded()


class TokenCakeConnector(OffloadingConnector):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        settings = vllm_config._tokencake_config
        if settings is None or not settings.offload.enabled:
            raise ValueError(
                "TokenCakeConnector requires structured TokenCake offload enablement"
            )
        super().__init__(vllm_config, role, kv_cache_config)

    def _create_scheduler(self, spec: OffloadingSpec) -> OffloadingConnectorScheduler:
        if not isinstance(spec, CPUOffloadingSpec):
            raise ValueError("TokenCake requires a native CPU offloading spec")
        if spec.num_blocks <= 0:
            raise ValueError(
                "TokenCake CPU offload capacity must fit at least one KV block"
            )
        return TokenCakeOffloadingScheduler(spec)

    @property
    def tokencake_scheduler(self) -> TokenCakeOffloadingScheduler:
        assert isinstance(self.connector_scheduler, TokenCakeOffloadingScheduler)
        return self.connector_scheduler

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        self.tokencake_scheduler.block_pool = gpu_block_pool

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        self.tokencake_scheduler.generation_finished(request, block_ids)
        return super().request_finished_all_groups(request, block_ids)
