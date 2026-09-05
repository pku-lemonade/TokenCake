# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any
from uuid import uuid1, uuid4

import pytest
from pydantic import ValidationError

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.tokencake.protocol import (
    metadata_from_extra_args,
    validate_generation_count,
)


@pytest.fixture(params=[CompletionRequest, ChatCompletionRequest, ResponsesRequest])
def request_factory(request):
    cls = request.param

    def create(**kwargs):
        inputs: dict[str, Any] = {"model": "model"}
        if cls is CompletionRequest:
            inputs["prompt"] = "hello"
        elif cls is ChatCompletionRequest:
            inputs["messages"] = [{"role": "user", "content": "hello"}]
        else:
            inputs["input"] = "hello"
        return cls(**(inputs | kwargs))

    return create


def test_recursive_json_and_existing_extensions(request_factory):
    values = {
        "string": "custom",
        "integer": 1,
        "number": 1.5,
        "list": ["custom", 2, 2.5],
        "boolean": True,
        "nested": {"list": [None, {"key": [1, False]}]},
    }
    request = request_factory(vllm_xargs=values)
    assert request.vllm_xargs == values
    params = request.to_sampling_params(16, {})
    assert params.extra_args == values
    assert metadata_from_extra_args(params.extra_args) is None


def test_partial_metadata_survives_transport(request_factory):
    lifecycle_id = f"tc-{uuid4().hex}"
    unknown = {"future": [False, {"nested": [1, None]}]}
    values = {"lifecycle_id": lifecycle_id, "depth": 3, "custom": unknown}
    request = request_factory(
        request_id=lifecycle_id,
        vllm_xargs={"tokencake": values, "ordinary": 2},
        kv_transfer_params={"remote_engine_id": "native"},
    )
    assert request.vllm_xargs["tokencake"] == values
    params = request.to_sampling_params(16, {})
    metadata = metadata_from_extra_args(params.extra_args)
    assert metadata is not None
    assert metadata.lifecycle_id == lifecycle_id and metadata.depth == 3
    assert metadata.importance == 0 and metadata.memory_weight == 1
    assert not metadata.offload_eligible and not metadata.reusable_prefix
    assert metadata.model_extra["custom"] == unknown
    assert params.extra_args["ordinary"] == 2
    assert params.extra_args["kv_transfer_params"] == {"remote_engine_id": "native"}


@pytest.mark.parametrize(
    "field,value",
    [
        ("agent_type", None),
        ("agent_name", 2),
        ("importance", "1"),
        ("importance", -1),
        ("importance", True),
        ("depth", 1.5),
        ("in_degree", -1),
        ("out_degree", False),
        ("similarity", 1.1),
        ("application_started_at_s", float("nan")),
        ("application_elapsed_s", -1),
        ("application_start_offset_s", -1),
        ("critical_path", 1),
        ("near_completion", "true"),
        ("join_group", []),
        ("dependency_depth", -1),
        ("fanout_width", 0),
        ("memory_weight", -1),
        ("offload_eligible", "yes"),
        ("reusable_prefix", 1),
    ],
)
def test_reject_malformed_known_metadata(request_factory, field, value):
    lifecycle_id = f"tc-{uuid4().hex}"
    with pytest.raises(ValidationError):
        request_factory(
            request_id=lifecycle_id,
            vllm_xargs={"tokencake": {"lifecycle_id": lifecycle_id, field: value}},
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        False,
        {},
        {"lifecycle_id": "tc-invalid"},
        {"lifecycle_id": f"tc-{uuid1().hex}"},
    ],
)
def test_reject_invalid_namespace(request_factory, value):
    with pytest.raises(ValidationError):
        request_factory(request_id=f"tc-{uuid4().hex}", vllm_xargs={"tokencake": value})


def test_mismatched_id_and_depths(request_factory):
    lifecycle_id = f"tc-{uuid4().hex}"
    with pytest.raises(ValidationError, match="match"):
        request_factory(
            request_id=f"tc-{uuid4().hex}",
            vllm_xargs={"tokencake": {"lifecycle_id": lifecycle_id}},
        )
    with pytest.raises(ValidationError, match="application_max_depth"):
        request_factory(
            request_id=lifecycle_id,
            vllm_xargs={
                "tokencake": {
                    "lifecycle_id": lifecycle_id,
                    "depth": 3,
                    "application_max_depth": 2,
                }
            },
        )


@pytest.mark.parametrize("cls", [CompletionRequest, ChatCompletionRequest])
@pytest.mark.parametrize("choices", [{"n": 2}, {"use_beam_search": True}])
def test_reject_annotated_choice_expansion(cls, choices):
    lifecycle_id = f"tc-{uuid4().hex}"
    inputs: dict[str, Any] = (
        {"prompt": "hello"}
        if cls is CompletionRequest
        else {"messages": [{"role": "user", "content": "hello"}]}
    )
    with pytest.raises(ValidationError, match="exactly one"):
        cls(
            **inputs,
            **choices,
            request_id=lifecycle_id,
            vllm_xargs={"tokencake": {"lifecycle_id": lifecycle_id}},
        )
    assert cls(**inputs, **choices)


def test_chat_null_n_is_a_single_generation():
    lifecycle_id = f"tc-{uuid4().hex}"
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}],
        n=None,
        request_id=lifecycle_id,
        vllm_xargs={"tokencake": {"lifecycle_id": lifecycle_id}},
    )
    assert request.to_sampling_params(16, {}).n == 1


def test_expanded_input_count_and_builtin_tools():
    args = {"tokencake": {"lifecycle_id": f"tc-{uuid4().hex}"}}
    validate_generation_count(args, 1)
    for count in (0, 2, 3):
        with pytest.raises(ValueError, match="exactly one"):
            validate_generation_count(args, count)
        validate_generation_count(None, count)
    with pytest.raises(ValueError, match="exactly one"):
        validate_generation_count(args, 1, builtin_tools=True)
    validate_generation_count(None, 1, builtin_tools=True)
