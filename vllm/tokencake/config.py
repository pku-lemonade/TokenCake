# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict, immutable settings for the additional_config TokenCake namespace."""

import math
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, StrictBool, StrictFloat, StrictInt, model_validator
from typing_extensions import Self

from vllm.config.utils import config

if TYPE_CHECKING:
    from vllm.config import VllmConfig

Ratio = Annotated[StrictFloat, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
PositiveRatio = Annotated[StrictFloat, Field(gt=0.0, le=1.0, allow_inf_nan=False)]
NonnegativeFloat = Annotated[StrictFloat, Field(ge=0.0, allow_inf_nan=False)]
PositiveFloat = Annotated[StrictFloat, Field(gt=0.0, allow_inf_nan=False)]
TemporalSelection = Literal["first_fit", "best_fit", "priority_first"]


@config(frozen=True)
class SchedulingConfig:
    enabled: StrictBool = True
    """Enable agent-aware scheduling."""
    temporal_selection: TemporalSelection = "first_fit"
    """Selection order for eligible preservation windows."""
    critical_ratio: PositiveRatio = 0.75
    """Fraction of agent types receiving reserved capacity, ranked by importance."""
    reserve_ratio_min: Ratio = 0.05
    """Minimum reserved fraction of physical KV capacity."""
    reserve_ratio_max: Ratio = 0.30
    """Maximum reserved fraction of physical KV capacity."""
    gpu_usage_low: Ratio = 0.40
    """Usage watermark below which reservation shrinks."""
    gpu_usage_high: Ratio = 0.75
    """Usage watermark above which reservation grows."""
    reserve_adjustment_step: PositiveRatio = 0.05
    """Bounded adjustment to the reserved fraction."""

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.reserve_ratio_min > self.reserve_ratio_max:
            raise ValueError(
                "tokencake reserve_ratio_min must not exceed reserve_ratio_max"
            )
        if self.gpu_usage_low >= self.gpu_usage_high:
            raise ValueError("tokencake gpu_usage_low must be below gpu_usage_high")
        return self


@config(frozen=True)
class TransferConfig:
    d2h_bandwidth_gbps: PositiveFloat = 12.0
    """Initial device-to-host bandwidth in GB/s."""
    h2d_bandwidth_gbps: PositiveFloat = 12.0
    """Initial host-to-device bandwidth in GB/s."""
    d2h_base_time_s: NonnegativeFloat = 0.0
    """Fixed device-to-host transfer cost in seconds."""
    h2d_base_time_s: NonnegativeFloat = 0.0
    """Fixed host-to-device transfer cost in seconds."""
    submission_time_per_run_s: NonnegativeFloat = 0.000003
    """Submission overhead for each contiguous copy run."""


@config(frozen=True)
class OffloadConfig:
    enabled: StrictBool = True
    """Enable lifecycle-based prefix preservation."""
    min_gpu_usage: Ratio = 0.60
    """Minimum GPU usage for proactive preservation."""
    high_pressure_gpu_usage: Ratio = 0.85
    """GPU usage watermark for high-pressure decisions."""
    score_threshold: NonnegativeFloat = 1.0
    """Minimum benefit-to-cost score for preservation."""
    backoff_steps: Annotated[StrictInt, Field(ge=0)] = 8
    """Scheduler steps to wait before reconsidering a rejected window."""
    eviction_window_blocks: Annotated[StrictInt, Field(gt=0)] = 128
    """Maximum number of entries inspected in the eviction window."""
    max_relief_blocks: Annotated[StrictInt, Field(gt=0)] = 32
    """Maximum snapshot blocks selected per preservation decision."""
    default_stall_s: PositiveFloat = 1.0
    """Default estimate when the caller supplies no stall duration."""
    release_lead_s: NonnegativeFloat = 0.10
    """Minimum lead time for predicted CPU ownership release."""
    ewma_alpha: PositiveRatio = 0.5
    """Weight assigned to each new duration sample."""
    transfer: TransferConfig = Field(default_factory=TransferConfig)
    """Initial transfer-cost estimates."""

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.min_gpu_usage >= self.high_pressure_gpu_usage:
            raise ValueError(
                "tokencake min_gpu_usage must be below high_pressure_gpu_usage"
            )
        return self


@config(frozen=True)
class TokenCakeConfig:
    scheduling: SchedulingConfig = Field(default_factory=SchedulingConfig)
    """Independent agent-scheduling settings."""
    offload: OffloadConfig = Field(default_factory=OffloadConfig)
    """Independent lifecycle-offload settings."""

    @property
    def temporal_selection(self) -> TemporalSelection:
        return (
            self.scheduling.temporal_selection
            if self.scheduling.enabled
            else "first_fit"
        )


def parse_tokencake_config(additional_config: object) -> TokenCakeConfig | None:
    if not isinstance(additional_config, dict) or "tokencake" not in additional_config:
        return None
    value = additional_config["tokencake"]
    if not isinstance(value, dict):
        raise ValueError("additional_config['tokencake'] must be an object")
    for section_name in ("scheduling", "offload"):
        section = value.get(section_name)
        if (
            isinstance(section, dict)
            and section.get("enabled") is False
            and section.keys() - {"enabled"}
        ):
            raise ValueError(
                f"tokencake.{section_name} is disabled but supplies tuning fields"
            )
    return TokenCakeConfig(**value)


def validate_prerequisites(
    vllm_config: "VllmConfig", settings: TokenCakeConfig
) -> None:
    if not (settings.scheduling.enabled or settings.offload.enabled):
        return
    if vllm_config.parallel_config.data_parallel_size > 1:
        raise ValueError("TokenCake requires data_parallel_size=1")
    if not settings.offload.enabled:
        return

    from vllm import envs

    cache = vllm_config.cache_config
    capacity = cache.kv_offloading_size
    if capacity is None or not math.isfinite(capacity) or capacity <= 0:
        raise ValueError(
            "TokenCake offload requires positive kv_offloading_size in GiB"
        )
    if cache.kv_offloading_backend != "native":
        raise ValueError("TokenCake offload requires kv_offloading_backend='native'")
    if envs.VLLM_USE_SIMPLE_KV_OFFLOAD:
        raise ValueError(
            "TokenCake requires OffloadingConnector; set VLLM_USE_SIMPLE_KV_OFFLOAD=0"
        )
    transfer = vllm_config.kv_transfer_config
    if transfer is None:
        return
    if transfer.kv_connector not in (None, "OffloadingConnector", "TokenCakeConnector"):
        raise ValueError(
            "TokenCake offload is incompatible with the configured connector"
        )
    if transfer.kv_connector_module_path is not None:
        raise ValueError("TokenCake offload requires the registered native connector")
    if (
        transfer.kv_connector_extra_config.get("spec_name", "CPUOffloadingSpec")
        != "CPUOffloadingSpec"
    ):
        raise ValueError(
            "TokenCake offload requires the native CPUOffloadingSpec manager"
        )
