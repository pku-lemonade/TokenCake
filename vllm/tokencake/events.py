# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict HTTP event models and typed EngineCore utility messages."""

from typing import Annotated, Literal

import msgspec
from pydantic import BaseModel, ConfigDict, Field, field_validator

from vllm.tokencake.protocol import LifecycleId

EventName = Literal["stall_started", "stall_finished"]
Disposition = Literal[
    "applied", "duplicate", "late_finish", "unknown", "conflict", "unavailable"
]


class StallStarted(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    event: Literal["stall_started"]
    lifecycle_id: LifecycleId
    kind: Annotated[str, Field(min_length=1)] = "stall"
    estimated_duration_s: Annotated[float, Field(gt=0)] | None = None

    @field_validator("estimated_duration_s", mode="before")
    @classmethod
    def reject_null_estimate(cls, value):
        if value is None:
            raise ValueError("estimated_duration_s must be a positive number")
        return value


class StallFinished(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    event: Literal["stall_finished"]
    lifecycle_id: LifecycleId


class LifecycleEvent(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    event: EventName
    lifecycle_id: str
    kind: str = "stall"
    estimated_duration_s: float | None = None


class LifecycleEventResult(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    lifecycle_id: str
    event: EventName
    state: str
    disposition: Disposition
    status_code: int = 200
