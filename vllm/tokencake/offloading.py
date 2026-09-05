# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TokenCake's scheduler-side extension of native CPU offload."""

from copy import deepcopy
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.tokencake.lifecycle import (
    GenerationCause,
    LifecycleRegistry,
    PrefixSnapshot,
    SnapshotGroup,
)
from vllm.tokencake.metrics import Metric
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import OffloadingSpec, ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.request import Request, RequestStatus


class TokenCakeOffloadingScheduler(OffloadingConnectorScheduler):
    def __init__(self, spec: CPUOffloadingSpec) -> None:
        super().__init__(spec)
        if not isinstance(self.manager, CPUOffloadingManager):
            raise ValueError("TokenCake requires the concrete CPUOffloadingManager")
        settings = spec.vllm_config._tokencake_config
        assert settings is not None and settings.offload.enabled
        self.lifecycles = LifecycleRegistry(settings.offload, self.manager)
        self.block_pool: BlockPool | None = None

    def _defer_store(self, request: Request) -> bool:
        params = request.sampling_params
        return params is not None and "tokencake" in (params.extra_args or {})

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

    def evaluate_pending(self) -> None:
        self.lifecycles.expire()
        for record in self.lifecycles.records.values():
            if record.pending_evaluation:
                record.pending_evaluation = False
                if (
                    not record.metadata.offload_eligible
                    or not record.metadata.reusable_prefix
                ):
                    self.lifecycles.metrics.count(Metric.NOT_ELIGIBLE)
                elif record.snapshot is None:
                    self.lifecycles.metrics.count(Metric.EMPTY)

    def reset_cache(self) -> None:
        super().reset_cache()
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
