# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TokenCake's scheduler-side extension of native CPU offload."""

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import OffloadingSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec


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
        scheduler = super()._create_scheduler(spec)
        if not isinstance(scheduler.manager, CPUOffloadingManager):
            raise ValueError("TokenCake requires the concrete CPUOffloadingManager")
        return scheduler
