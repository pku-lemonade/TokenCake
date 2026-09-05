# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.config import LoRAConfig
from vllm.config.utils import replace
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import PlaceholderRange
from vllm.tokencake.protocol import TokenCakeMetadata
from vllm.tokencake.scheduling import (
    AgentHistory,
    CapacityPlan,
    partition_capacity,
    request_score,
)
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheGroupSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus


def make_scheduler(*, enabled=True, policy="fcfs", groups=1, **kwargs):
    base = create_scheduler(**kwargs)
    config = replace(
        base.vllm_config,
        scheduler_config=replace(
            base.scheduler_config,
            policy=policy,
            max_model_len=base.max_model_len,
            is_encoder_decoder=base.is_encoder_decoder,
        ),
        additional_config={
            "tokencake": {
                "scheduling": {"enabled": enabled},
                "offload": {"enabled": False},
            }
        },
    )
    cache = base.kv_cache_config
    if groups == 2:
        group = cache.kv_cache_groups[0]
        cache = replace(
            cache,
            kv_cache_groups=[
                group,
                KVCacheGroupSpec(
                    ["other"], replace(group.kv_cache_spec, block_size=32)
                ),
            ],
        )
    return Scheduler(
        config,
        cache,
        base.structured_output_manager,
        base.cache_config.block_size,
        log_stats=True,
    )


def make_request(name, *, importance=0.0, agent_type="agent", tokens=32, priority=0):
    request = create_requests(1, num_tokens=tokens, req_ids=[name], max_tokens=128)[0]
    request.priority = priority
    request.arrival_time = 1000.0
    if importance is not None:
        request.sampling_params = copy.deepcopy(request.sampling_params)
        request.sampling_params.extra_args = {
            "tokencake": {
                "lifecycle_id": f"tc-{uuid4().hex}",
                "agent_type": agent_type,
                "importance": float(importance),
            }
        }
    return request


def decode(scheduler, output):
    ids = list(output.num_scheduled_tokens)
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={key: i for i, key in enumerate(ids)},
            sampled_token_ids=[[100] for _ in ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def test_frozen_source_scores_and_reservation_partition():
    golden = json.loads(
        Path(__file__).with_name("scheduling_reference.json").read_text()
    )
    assert golden["source_revision"] == "7a608a4e53ea990b2540c93b4d28cb795b905109"
    for case in golden["requests"]:
        metadata = TokenCakeMetadata(
            lifecycle_id=f"tc-{uuid4().hex}", **case["metadata"]
        )
        assert request_score(metadata, case["arrival"], case["now"]) == case["score"]
    for key, values in golden["histories"].items():
        assert AgentHistory(**values).score(
            values["importance"], golden["waiting"][key], golden["average_wait"][key]
        ) == pytest.approx(golden["scores"][key])
    plan = partition_capacity(
        golden["scores"],
        {key: values["importance"] for key, values in golden["histories"].items()},
        golden["used"],
        1000,
        golden["reserve_ratio"],
        0.75,
    )
    assert plan.critical == set(golden["critical"])
    assert plan.reserved == golden["reserved"]
    assert plan.shared == golden["shared"]


@pytest.mark.parametrize("policy", ["fcfs", "priority"])
@pytest.mark.parametrize("split", [0, 1, 2, 3])
def test_native_ordinary_barrier_in_both_queues(policy, split):
    scheduler = make_scheduler(policy=policy, max_num_seqs=1)
    requests = [
        make_request("a", importance=1),
        make_request("b", importance=None),
        make_request("c", importance=100),
    ]
    for i, request in enumerate(requests):
        request.arrival_time += i
        scheduler.add_request(request)
        if i < split:
            scheduler.waiting.remove_request(request)
            scheduler.skipped_waiting.add_request(request)
    for expected in requests:
        output = scheduler.schedule()
        assert list(output.num_scheduled_tokens) == [expected.request_id]
        scheduler.finish_requests(expected.request_id, RequestStatus.FINISHED_STOPPED)


@pytest.mark.parametrize("policy", ["fcfs", "priority"])
def test_frozen_score_exact_removal_and_native_tie_break(policy):
    scheduler = make_scheduler(policy=policy, max_num_seqs=1)
    a = make_request("a", importance=1, priority=2)
    b = make_request("b", importance=100, priority=1)
    c = make_request("c", importance=100, priority=0)
    for request in [a, b, c]:
        scheduler.add_request(request)
    controller = scheduler._tokencake_scheduling
    with patch("vllm.tokencake.scheduling.time.time", return_value=1000):
        controller.begin_step(scheduler.waiting, scheduler.running)
    frozen = dict(controller.scores)
    with patch("vllm.tokencake.scheduling.time.time", return_value=1300):
        assert min([a, b, c], key=controller.order_key) is c
    assert controller.scores == frozen
    assert list(scheduler.schedule().num_scheduled_tokens) == ["c"]
    assert set(r.request_id for r in scheduler.waiting) == {"a", "b"}


def test_native_blocked_handling_precedes_agent_selection():
    scheduler = make_scheduler(max_num_seqs=1)
    low = make_request("low", importance=1)
    blocked = make_request("blocked", importance=100)
    blocked.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    for request in [low, blocked]:
        scheduler.add_request(request)
    assert list(scheduler.schedule().num_scheduled_tokens) == ["low"]
    assert blocked in scheduler.skipped_waiting


def test_lora_eligibility_precedes_agent_selection():
    scheduler = make_scheduler(max_num_seqs=4)
    scheduler.lora_config = LoRAConfig(max_loras=1)
    active = make_request("active")
    active.lora_request = LoRARequest("one", 1, "/unused/one")
    scheduler.add_request(active)
    decode(scheduler, scheduler.schedule())
    eligible = make_request("eligible", importance=1)
    eligible.lora_request = active.lora_request
    blocked = make_request("blocked", importance=100)
    blocked.lora_request = LoRARequest("two", 2, "/unused/two")
    for request in [eligible, blocked]:
        scheduler.add_request(request)
    output = scheduler.schedule()
    assert set(output.num_scheduled_tokens) == {"active", "eligible"}
    assert blocked in scheduler.skipped_waiting


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("annotated", [False, True])
@pytest.mark.parametrize("threshold", [128, 1024])
def test_prefill_cap_preserves_native_clamps(enabled, annotated, threshold):
    scheduler = make_scheduler(enabled=enabled, long_prefill_token_threshold=threshold)
    request = make_request("long", importance=1 if annotated else None, tokens=1024)
    scheduler.add_request(request)
    output = scheduler.schedule()
    expected = min(threshold, 256) if enabled and annotated else threshold
    assert output.num_scheduled_tokens["long"] == expected
    assert request.status == RequestStatus.RUNNING


@pytest.mark.parametrize("groups,expected", [(1, 2), (2, 3)])
def test_actual_multi_group_allocation_commit_and_exact_release(groups, expected):
    scheduler = make_scheduler(groups=groups)
    request = make_request("request")
    scheduler.add_request(request)
    controller = scheduler._tokencake_scheduling
    pool = scheduler.kv_cache_manager.block_pool
    free = pool.get_num_free_blocks()
    scheduler.schedule()
    assert free - pool.get_num_free_blocks() == expected
    assert len(controller.charges) == expected
    scheduler.finish_requests("request", RequestStatus.FINISHED_ABORTED)
    assert not controller.charges
    assert not controller.metadata and not controller.started
    assert pool.get_num_free_blocks() == free
    scheduler.finish_requests("request", RequestStatus.FINISHED_ABORTED)
    assert pool.get_num_free_blocks() == free


def test_prefix_sharing_keeps_charge_until_last_native_reference():
    scheduler = make_scheduler(enable_prefix_caching=True)
    first = make_request("first", tokens=32)
    second = make_request("second", tokens=31)
    scheduler.add_request(first)
    decode(scheduler, scheduler.schedule())
    scheduler.add_request(second)
    scheduler.schedule()
    controller = scheduler._tokencake_scheduling
    manager = scheduler.kv_cache_manager
    shared = manager.get_blocks("second").blocks[0][0]
    assert shared.ref_cnt == 2
    scheduler.finish_requests("first", RequestStatus.FINISHED_STOPPED)
    assert shared.ref_cnt == 1 and shared.block_id in controller.charges
    scheduler.finish_requests("second", RequestStatus.FINISHED_STOPPED)
    assert shared.ref_cnt == 0 and not controller.charges


def test_ordinary_shared_only_and_annotated_idle_borrowing():
    scheduler = make_scheduler()
    controller = scheduler._tokencake_scheduling
    owner = make_request("owner", agent_type="owner")
    other = make_request("other", agent_type="other")
    ordinary = make_request("ordinary", importance=None)
    for request in [owner, other, ordinary]:
        scheduler.add_request(request)
    controller.begin_step(scheduler.waiting, [])
    controller.plan = CapacityPlan({"owner"}, {"owner": 5.0}, {"owner": 10}, 5)
    controller.shared_available = 2
    controller.reserved_available = {"owner": 10}
    controller.waiting_critical = {"owner"}
    assert controller._consumption(ordinary, 2) == [(None, 2)]
    assert controller._consumption(ordinary, 3) is None
    assert controller._consumption(other, 3) is None
    assert controller._consumption(owner, 3) == [(None, 2), ("owner", 1)]
    controller.waiting_critical.clear()
    assert controller._consumption(other, 3) == [(None, 2), ("owner", 1)]
    assert controller._consumption(ordinary, 3) is None
    assert controller.shared_available == 2
    assert controller.reserved_available == {"owner": 10}


def test_reservation_denial_and_allocation_failure_leave_request_pending():
    scheduler = make_scheduler(num_blocks=101)
    ordinary = make_request("ordinary", importance=None)
    owner = make_request("owner", tokens=160)
    scheduler.add_request(ordinary)
    scheduler.add_request(owner)
    controller = scheduler._tokencake_scheduling
    controller.plan = CapacityPlan({"agent"}, {"agent": 5.0}, {"agent": 100}, 0)
    output = scheduler.schedule()
    assert "ordinary" not in output.num_scheduled_tokens
    assert ordinary in scheduler.skipped_waiting
    assert not ordinary.is_finished()
    assert "owner" in output.num_scheduled_tokens
    assert set(controller.charges.values()) == {"agent"}
    scheduler.finish_requests("owner", RequestStatus.FINISHED_STOPPED)
    assert not controller.charges
    assert "ordinary" in scheduler.schedule().num_scheduled_tokens

    scheduler = make_scheduler(num_blocks=2)
    oversized = make_request("oversized", tokens=64)
    scheduler.add_request(oversized)
    assert not scheduler.schedule().num_scheduled_tokens
    assert oversized in scheduler.waiting
    assert not scheduler._tokencake_scheduling.charges


@pytest.mark.parametrize("victim_scheduled", [False, True])
@pytest.mark.parametrize("margin", [10.0, 100.0])
def test_native_preemption_rollback_and_recomputation(victim_scheduled, margin):
    scheduler = make_scheduler(num_blocks=6, max_num_batched_tokens=200)
    low = make_request("low", importance=0)
    high = make_request("high", importance=margin)
    scheduler.add_request(low)
    decode(scheduler, scheduler.schedule())
    scheduler.add_request(high)
    decode(scheduler, scheduler.schedule())
    if not victim_scheduled:
        scheduler.running.reverse()
    output = scheduler.schedule()
    victim = low if margin == 100 else high
    survivor = high if victim is low else low
    assert victim.status == RequestStatus.PREEMPTED
    assert victim in scheduler.waiting and not victim.is_finished()
    assert victim.num_computed_tokens == 0
    assert victim.request_id not in output.num_scheduled_tokens
    assert victim.request_id not in output.scheduled_spec_decode_tokens
    assert victim.request_id not in output.scheduled_encoder_inputs
    assert survivor in scheduler.running
    assert sum(output.num_scheduled_tokens.values()) <= 200
    controller = scheduler._tokencake_scheduling
    assert len(controller.charges) == len(controller._block_ids(survivor))
    scheduler.finish_requests(survivor.request_id, RequestStatus.FINISHED_STOPPED)
    resumed = scheduler.schedule()
    assert victim.request_id in resumed.num_scheduled_tokens
    assert victim.status == RequestStatus.RUNNING
    scheduler.finish_requests(victim.request_id, RequestStatus.FINISHED_STOPPED)
    assert not controller.charges and not controller.metadata


@pytest.mark.parametrize("policy", ["fcfs", "priority"])
def test_running_reservation_denial_requeues_to_unblock_waiting_owner(policy):
    scheduler = make_scheduler(num_blocks=11, policy=policy)
    running = make_request("running", agent_type="borrower", tokens=128)
    owner = make_request("owner", agent_type="owner", tokens=64, importance=100)
    scheduler.add_request(running)
    scheduler.waiting.remove_request(running)
    scheduler.running.append(running)
    running.status = RequestStatus.RUNNING
    assert scheduler.kv_cache_manager.allocate_slots(running, 128) is not None
    running.num_computed_tokens = 128
    running.append_output_token_ids([100])
    scheduler.add_request(owner)
    controller = scheduler._tokencake_scheduling
    controller.plan = CapacityPlan({"owner"}, {"owner": 100.0}, {"owner": 2}, 8)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 2
    output = scheduler.schedule()
    assert running.status == RequestStatus.PREEMPTED
    assert running in scheduler.waiting and not running.is_finished()
    assert not output.num_scheduled_tokens
    resumed = scheduler.schedule()
    assert "owner" in resumed.num_scheduled_tokens
    scheduler.finish_requests("owner", RequestStatus.FINISHED_STOPPED)
    assert "running" in scheduler.schedule().num_scheduled_tokens


def test_scheduled_victim_restores_encoder_speculative_and_token_budgets():
    scheduler = make_scheduler(num_blocks=7, max_num_batched_tokens=64)
    scheduler.max_num_encoder_input_tokens = 8
    scheduler.encoder_cache_manager = EncoderCacheManager(32)
    low = make_request("low", tokens=32)
    high = make_request("high", tokens=48, importance=100)
    following = make_request("following", tokens=16, importance=None)
    for request, allocated, computed, offset in [
        (low, 48, 16, 20),
        (high, 32, 32, 40),
        (following, 16, 0, 4),
    ]:
        request.mm_features = create_requests(
            1,
            num_tokens=request.num_prompt_tokens,
            mm_hashes_list=[[request.request_id]],
            mm_positions=[[PlaceholderRange(offset=offset, length=4)]],
        )[0].mm_features
        scheduler.add_request(request)
        scheduler.waiting.remove_request(request)
        scheduler.running.append(request)
        request.status = RequestStatus.RUNNING
        assert scheduler.kv_cache_manager.allocate_slots(request, allocated) is not None
        request.num_computed_tokens = computed
    low.spec_token_ids = [200, 201]
    output = scheduler.schedule()
    assert low.status == RequestStatus.PREEMPTED and low in scheduler.waiting
    assert output.num_scheduled_tokens == {"high": 16, "following": 16}
    assert output.scheduled_encoder_inputs == {"high": [0], "following": [0]}
    assert not output.scheduled_spec_decode_tokens
    assert not low.spec_token_ids
    assert not scheduler.encoder_cache_manager.cached["low"]
    assert len(scheduler._tokencake_scheduling.charges) == 4


def test_pressure_adjustment_bounds_and_fixed_interval():
    scheduler = make_scheduler(num_blocks=101)
    request = make_request("request", tokens=1440)
    scheduler.add_request(request)
    controller = scheduler._tokencake_scheduling
    assert scheduler.kv_cache_manager.allocate_slots(request, 1440) is not None
    controller.begin_step([request], [])
    assert controller.reserve_ratio == pytest.approx(0.10)
    controller.begin_step([request], [])
    assert controller.reserve_ratio == pytest.approx(0.10)
    for step in range(499, 3500, 500):
        controller.step = step
        controller.begin_step([request], [])
    assert controller.reserve_ratio == 0.30
    scheduler.kv_cache_manager.free(request)
    for step in range(3999, 7500, 500):
        controller.step = step
        controller.begin_step([request], [])
    assert controller.reserve_ratio == 0.05


@pytest.mark.parametrize("policy", ["fcfs", "priority"])
@pytest.mark.parametrize("enabled", [False, True])
def test_all_ordinary_uses_native_path_without_scores_or_queue_scan(policy, enabled):
    scheduler = make_scheduler(enabled=enabled, policy=policy)
    for request in [
        make_request("a", importance=None),
        make_request("b", importance=None),
    ]:
        scheduler.add_request(request)
    assert type(scheduler.waiting) is type(
        create_request_queue(SchedulingPolicy(policy))
    )
    controller = scheduler._tokencake_scheduling
    if not enabled:
        assert controller is None
    with (
        patch("vllm.tokencake.scheduling.request_score", side_effect=AssertionError),
        patch(
            "vllm.tokencake.scheduling.SchedulingController.annotated_prefix",
            side_effect=AssertionError,
        ),
    ):
        assert list(scheduler.schedule().num_scheduled_tokens) == ["a", "b"]
    assert scheduler.connector is None
