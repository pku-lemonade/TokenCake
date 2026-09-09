# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation for the extensible TokenCake request metadata namespace."""

from collections.abc import Mapping
from typing import Annotated, Any
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    model_validator,
)
from typing_extensions import Self


def validate_lifecycle_id(value: str) -> str:
    message = "TokenCake lifecycle_id must be tc- followed by canonical UUID4 hex"
    if len(value) != 35 or not value.startswith("tc-"):
        raise ValueError(message)
    try:
        identifier = UUID(hex=value[3:])
    except ValueError as exc:
        raise ValueError(message) from exc
    if identifier.version != 4 or identifier.hex != value[3:]:
        raise ValueError(message)
    return value


LifecycleId = Annotated[str, Field(strict=True), AfterValidator(validate_lifecycle_id)]
NonnegativeInt = Annotated[int, Field(ge=0)]
NonnegativeFloat = Annotated[float, Field(ge=0)]


class TokenCakeMetadata(BaseModel):
    model_config = ConfigDict(
        extra="allow", frozen=True, strict=True, allow_inf_nan=False
    )

    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)
    lifecycle_id: LifecycleId
    agent_type: str = ""
    agent_name: str = ""
    importance: NonnegativeFloat = 0.0
    depth: NonnegativeInt = 0
    in_degree: NonnegativeInt = 0
    out_degree: NonnegativeInt = 0
    similarity: Annotated[float, Field(ge=0, le=1)] = 0.0
    application_started_at_s: NonnegativeFloat = 0.0
    application_start_offset_s: NonnegativeFloat = 0.0
    application_elapsed_s: NonnegativeFloat = 0.0
    application_max_depth: NonnegativeInt = 0
    remaining_depth: NonnegativeInt = 0
    critical_path: bool = False
    near_completion: bool = False
    join_group: str = ""
    dependency_depth: NonnegativeInt = 0
    fanout_width: Annotated[int, Field(ge=1)] = 1
    memory_weight: NonnegativeFloat = 1.0
    offload_eligible: bool = False
    reusable_prefix: bool = False

    @model_validator(mode="after")
    def validate_depths(self) -> Self:
        if "application_max_depth" in self.model_fields_set:
            if self.depth > self.application_max_depth:
                raise ValueError("TokenCake depth exceeds application_max_depth")
            if self.remaining_depth > self.application_max_depth:
                raise ValueError(
                    "TokenCake remaining_depth exceeds application_max_depth"
                )
        return self


def metadata_from_extra_args(
    extra_args: Mapping[str, Any] | None,
) -> TokenCakeMetadata | None:
    if extra_args is None or "tokencake" not in extra_args:
        return None
    value = extra_args["tokencake"]
    if not isinstance(value, dict):
        raise ValueError("vllm_xargs['tokencake'] must be an object")
    return TokenCakeMetadata.model_validate(value)


def validate_request_metadata(
    extra_args: dict[str, JsonValue] | None,
    request_id: str,
    *,
    n: int | None = 1,
    use_beam_search: bool = False,
) -> None:
    metadata = metadata_from_extra_args(extra_args)
    if metadata is None:
        return
    if metadata.lifecycle_id != request_id:
        raise ValueError("TokenCake lifecycle_id must match the top-level request_id")
    if n not in (None, 1) or use_beam_search:
        raise ValueError(
            "TokenCake metadata requires exactly one EngineCore generation"
        )
    assert extra_args is not None
    extra_args["tokencake"] = metadata.model_dump(exclude_unset=True)


def validate_generation_count(
    extra_args: Mapping[str, Any] | None,
    count: int,
    *,
    builtin_tools: bool = False,
) -> None:
    if (
        extra_args is not None
        and "tokencake" in extra_args
        and (count != 1 or builtin_tools)
    ):
        raise ValueError(
            "TokenCake metadata requires exactly one EngineCore generation"
        )
