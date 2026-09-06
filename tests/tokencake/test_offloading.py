# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import weakref
from uuid import uuid4

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from tests.v1.kv_connector.unit.offloading_connector.utils import MockOffloadingHandler
from vllm.config import KVTransferConfig
from vllm.config.utils import replace
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.tokencake.events import LifecycleEvent
from vllm.tokencake.metrics import Metric
from vllm.tokencake.offload_policy import Pressure, evaluate_benefit, waiting_pressure
from vllm.tokencake.protocol import TokenCakeMetadata
from vllm.tokencake.scheduling import CapacityPlan
from vllm.v1.engine.core import EngineCore
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    KVCacheTensor,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.base import GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.worker.worker import TransferResult
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus


def make_offload_scheduler(
    *,
    gpu_blocks=17,
    cpu_blocks=64,
    factor=1,
    scheduling=False,
    groups=1,
    settings=None,
    cache_policy="lru",
    async_scheduling=False,
    max_num_batched_tokens=512,
    decode_prefill_token_budget=0,
    generation_reserve_mode="reclaim",
):
    base = create_scheduler(
        num_blocks=gpu_blocks,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_prefix_caching=True,
        async_scheduling=async_scheduling,
    )
    cache = base.kv_cache_config
    if groups == 2:
        spec = cache.kv_cache_groups[0].kv_cache_spec
        cache = replace(
            cache,
            kv_cache_groups=[
                cache.kv_cache_groups[0],
                KVCacheGroupSpec(["other"], replace(spec, block_size=32)),
            ],
        )
    elif groups == "hybrid":
        spec = cache.kv_cache_groups[0].kv_cache_spec
        cache = replace(
            cache,
            kv_cache_groups=[
                cache.kv_cache_groups[0],
                KVCacheGroupSpec(
                    ["other"],
                    SlidingWindowSpec(
                        block_size=16,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=spec.dtype,
                        sliding_window=32,
                    ),
                ),
            ],
        )
    cache = replace(
        cache,
        kv_cache_tensors=[
            KVCacheTensor(
                size=gpu_blocks * group.kv_cache_spec.page_size_bytes,
                shared_by=group.layer_names,
            )
            for group in cache.kv_cache_groups
        ],
    )
    bytes_per_block = sum(t.size for t in cache.kv_cache_tensors) // gpu_blocks * factor
    config = replace(
        base.vllm_config,
        cache_config=replace(
            base.cache_config,
            kv_offloading_size=cpu_blocks * bytes_per_block / 2**30,
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                **({"block_size": 16 * factor} if factor != 1 else {}),
                "eviction_policy": cache_policy,
            },
        ),
        additional_config={
            "tokencake": {
                "scheduling": {
                    "enabled": scheduling,
                    **(
                        {
                            "decode_prefill_token_budget": decode_prefill_token_budget,
                            "generation_reserve_mode": generation_reserve_mode,
                        }
                        if scheduling
                        else {}
                    ),
                },
                "offload": {"min_gpu_usage": 0.0, **(settings or {})},
            }
        },
    )
    config.cache_config.num_gpu_blocks = gpu_blocks
    return type(base)(config, cache, base.structured_output_manager, 16, log_stats=True)


@pytest.fixture(autouse=True)
def native_offload(monkeypatch):
    monkeypatch.setenv("VLLM_USE_SIMPLE_KV_OFFLOAD", "0")


def request_for(name, *, tokens=32, value=0, annotated=True):
    request = create_requests(
        value + 1,
        num_tokens=tokens,
        req_ids=[f"{name}-{i}" for i in range(value + 1)],
    )[-1]
    if annotated:
        request.sampling_params = copy.deepcopy(request.sampling_params)
        request.sampling_params.extra_args = {
            "tokencake": {
                "lifecycle_id": f"tc-{uuid4().hex}",
                "agent_type": "tool",
                "offload_eligible": True,
                "reusable_prefix": True,
            }
        }
    return request


def complete_generation(scheduler, request):
    scheduler.add_request(request)
    output = scheduler.schedule()
    assert 0 < output.num_scheduled_tokens[request.request_id] <= request.num_tokens
    assert request.num_computed_tokens == request.num_tokens
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_STOPPED)
    offload = scheduler.connector.tokencake_scheduler
    identifier = request.sampling_params.extra_args["tokencake"]["lifecycle_id"]
    return offload.lifecycles.records[identifier]


def start(scheduler, record):
    core = object.__new__(EngineCore)
    core.scheduler = scheduler
    return core.tokencake_event(
        LifecycleEvent(
            "stall_started",
            record.metadata.lifecycle_id,
            estimated_duration_s=30.0,
        )
    )


def worker_for(scheduler):
    spec = CPUOffloadingSpec(scheduler.vllm_config, scheduler.kv_cache_config)
    worker = OffloadingConnectorWorker(spec)
    handler = MockOffloadingHandler()
    worker.worker.register_handler(GPULoadStoreSpec, CPULoadStoreSpec, handler)
    worker.worker.register_handler(CPULoadStoreSpec, GPULoadStoreSpec, handler)
    return worker, handler


def complete_jobs(scheduler):
    offload = scheduler.connector.tokencake_scheduler
    if offload.has_unpublished:
        publication = scheduler.schedule()
        assert not publication.num_scheduled_tokens
        assert publication.kv_connector_metadata.store_jobs
    offload.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                {j: 1 for j in offload._detached}
            ),
        )
    )


@pytest.mark.parametrize("in_event", [False, True])
@pytest.mark.parametrize("async_scheduling", [False, True])
def test_exclusive_publication_native_deferral_and_reuse_fence(
    in_event, async_scheduling
):
    scheduler = make_offload_scheduler(gpu_blocks=5, async_scheduling=async_scheduling)
    creator = request_for("creator")
    record = complete_generation(scheduler, creator)
    offload = scheduler.connector.tokencake_scheduler
    waiter = request_for("waiter", tokens=64, value=1)
    scheduler.add_request(waiter)
    if in_event:
        assert start(scheduler, record).disposition == "applied"
        assert offload.has_unpublished
    else:
        offload.lifecycles.apply(
            LifecycleEvent(
                "stall_started",
                record.metadata.lifecycle_id,
                estimated_duration_s=30.0,
            )
        )
    reference = weakref.ref(creator)
    del creator
    assert reference() is None
    publication = scheduler.schedule()
    assert not publication.num_scheduled_tokens
    assert not scheduler.kv_cache_manager.get_blocks(waiter.request_id).blocks[0]
    meta = publication.kv_connector_metadata
    assert isinstance(meta, OffloadingConnectorMetadata)
    assert meta.store_jobs and not meta.jobs_to_flush
    sources = {bid for job in offload._detached.values() for bid in job.source_ids}
    assert sources <= offload._block_id_to_pending_jobs.keys()
    assert all(
        scheduler.kv_cache_manager.block_pool.blocks[bid].ref_cnt == 0
        for bid in sources
    )
    worker, handler = worker_for(scheduler)
    worker.handle_preemptions(meta)
    worker.start_kv_transfers(meta)
    worker.prepare_store_kv(meta)
    assert not handler.transfer_specs
    assert not record.retained
    assert all(
        offload.manager._policy.get(k).ref_cnt == -1 for k in record.pending_retention
    )
    following = scheduler.schedule()
    assert following.num_scheduled_tokens[waiter.request_id] == 64
    next_meta = following.kv_connector_metadata
    assert meta.store_jobs.keys() <= next_meta.jobs_to_flush
    worker.handle_preemptions(next_meta)
    worker.start_kv_transfers(next_meta)
    assert handler.flushed_jobs == set(meta.store_jobs)
    worker.get_finished(set())
    offload.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=worker.build_connector_worker_meta(),
        )
    )
    assert record.retained and not record.pending_retention
    assert not offload._detached and not offload._block_id_to_pending_jobs


def test_start_before_last_generation_completion_is_one_shot_without_spin():
    scheduler = make_offload_scheduler()
    creator = request_for("creator")
    scheduler.add_request(creator)
    offload = scheduler.connector.tokencake_scheduler
    record = offload.lifecycles.records[
        creator.sampling_params.extra_args["tokencake"]["lifecycle_id"]
    ]
    assert start(scheduler, record).status_code == 200
    output = scheduler.schedule()
    assert not output.kv_connector_metadata.store_jobs
    assert (
        offload._req_status[creator.request_id].group_states[0].next_stored_block_idx
        == 0
    )
    scheduler.finish_requests(creator.request_id, RequestStatus.FINISHED_STOPPED)
    assert record.pending_evaluation and scheduler.has_requests()
    idle = scheduler.schedule()
    assert not idle.num_scheduled_tokens
    assert not record.pending_evaluation and not offload.has_pending_work
    assert not scheduler.has_requests()
    scheduler.add_request(request_for("later", tokens=64, value=1))
    published = scheduler.schedule()
    assert published.kv_connector_metadata.store_jobs
    assert not published.num_scheduled_tokens


@pytest.mark.parametrize("release", ["finish", "expiry", "predicted", "reset"])
def test_release_during_d2h_does_not_resurrect_ownership(release):
    scheduler = make_offload_scheduler()
    offload = scheduler.connector.tokencake_scheduler
    now = [100.0]
    offload.lifecycles.clock = lambda: now[0]
    record = complete_generation(scheduler, request_for("creator"))
    scheduler.add_request(request_for("waiting", tokens=64, value=1))
    start(scheduler, record)
    publication = scheduler.schedule()
    jobs = set(publication.kv_connector_metadata.store_jobs)
    keys = set(record.pending_retention)
    assert jobs and keys
    if release == "finish":
        offload.lifecycles.apply(
            LifecycleEvent("stall_finished", record.metadata.lifecycle_id)
        )
    elif release == "expiry":
        now[0] = record.safety_deadline
        offload.lifecycles.expire()
    elif release == "predicted":
        now[0] = record.release_deadline
        offload.lifecycles.expire()
    else:
        assert scheduler.reset_connector_cache()
    offload.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata({job: 1 for job in jobs}),
        )
    )
    assert not record.retained and not record.pending_retention
    if release == "reset":
        assert all(offload.manager._policy.get(key) is None for key in keys)
    else:
        assert all(offload.manager._policy.get(key).ref_cnt == 0 for key in keys)


@pytest.mark.parametrize("cache_policy", ["lru", "arc"])
def test_pending_key_waiters_and_native_load_stack_references(cache_policy):
    scheduler = make_offload_scheduler(cache_policy=cache_policy)
    offload = scheduler.connector.tokencake_scheduler
    first = complete_generation(scheduler, request_for("first"))
    second = complete_generation(scheduler, request_for("second"))
    scheduler.add_request(request_for("waiting", tokens=64, value=1))
    start(scheduler, first)
    jobs = set(offload._detached)
    start(scheduler, second)
    assert set(offload._detached) == jobs
    assert first.pending_retention == second.pending_retention
    keys = set(first.pending_retention)
    complete_jobs(scheduler)
    assert first.retained == second.retained == keys
    context = first.snapshot.req_context
    offload.manager.prepare_load(keys, context)
    assert all(offload.manager._policy.get(key).ref_cnt == 3 for key in keys)
    for record in [first, first, second]:
        offload.lifecycles.apply(
            LifecycleEvent("stall_finished", record.metadata.lifecycle_id)
        )
    assert all(offload.manager._policy.get(key).ref_cnt == 1 for key in keys)
    offload.manager.complete_load(keys, context)
    assert all(offload.manager._policy.get(key).ref_cnt == 0 for key in keys)


def test_frontier_observation_is_bounded_and_cannot_select_external_entries(
    monkeypatch,
):
    scheduler = make_offload_scheduler()
    offload = scheduler.connector.tokencake_scheduler
    record = complete_generation(scheduler, request_for("creator"))
    pool = scheduler.kv_cache_manager.block_pool
    before = pool.free_block_queue.get_all_free_blocks()
    assert len(pool.peek_free_block_frontier(2)) == 2
    assert pool.free_block_queue.get_all_free_blocks() == before
    assert pool.peek_free_block_frontier(0) == ()
    expected = [c.key for c in offload._snapshot_candidates(record)]
    monkeypatch.setattr(
        pool, "peek_free_block_frontier", lambda limit: ((9999, b"external"),)
    )
    scheduler.add_request(request_for("waiting", tokens=64, value=1))
    start(scheduler, record)
    actual = set().union(*(s.keys for s in offload._jobs.values()))
    assert actual == set(expected)
    assert offload.lifecycles.metrics.snapshot(1)[Metric.EXTERNAL.value] == 1


@pytest.mark.parametrize("groups,factor", [(1, 1), (1, 2), (2, 1), ("hybrid", 1)])
def test_native_group_specs_and_maximum_relief(groups, factor):
    scheduler = make_offload_scheduler(
        gpu_blocks=65,
        groups=groups,
        factor=factor,
        settings={"max_relief_blocks": 4},
    )
    offload = scheduler.connector.tokencake_scheduler
    record = complete_generation(
        scheduler, request_for("creator", tokens=32 if groups == "hybrid" else 128)
    )
    scheduler.add_request(request_for("waiting", tokens=128, value=1))
    start(scheduler, record)
    assert offload._unpublished
    for job in offload._unpublished.values():
        src, dst = job.transfer_spec
        assert isinstance(src, GPULoadStoreSpec) and isinstance(dst, CPULoadStoreSpec)
        assert len(src.block_ids) <= 4
        assert sum(src.group_sizes) == len(src.block_ids)
        assert len(src.group_sizes) == len(scheduler.kv_cache_config.kv_cache_groups)
    complete_jobs(scheduler)
    assert record.retained


@pytest.mark.parametrize("groups,factor", [(1, 1), (1, 2), (2, 1)])
def test_window_can_preserve_and_restore_more_than_32_gpu_blocks(groups, factor):
    scheduler = make_offload_scheduler(
        gpu_blocks=129,
        cpu_blocks=128,
        groups=groups,
        factor=factor,
        max_num_batched_tokens=1024,
    )
    record = complete_generation(scheduler, request_for("creator", tokens=640))
    offload = scheduler.connector.tokencake_scheduler
    scheduler.add_request(request_for("waiting", tokens=1024, value=1))
    start(scheduler, record)
    sources = [bid for job in offload._detached.values() for bid in job.source_ids]
    assert len(sources) == (60 if groups == 2 else 40)
    complete_jobs(scheduler)
    resumed = request_for("resumed", tokens=656)
    matched, asynchronous = offload.get_num_new_matched_tokens(resumed, 0)
    assert matched == 640 and asynchronous


@pytest.mark.parametrize("groups,factor", [(1, 1), (1, 2), (2, 1)])
def test_completed_prefix_can_be_saved_while_a_peer_still_uses_it(groups, factor):
    scheduler = make_offload_scheduler(gpu_blocks=65, groups=groups, factor=factor)
    record = complete_generation(scheduler, request_for("creator", tokens=128))
    peer = request_for("peer", tokens=144)
    scheduler.add_request(peer)
    scheduler.schedule()
    offload = scheduler.connector.tokencake_scheduler
    pool = scheduler.kv_cache_manager.block_pool
    sources = {
        bid
        for group in record.snapshot.groups
        for ids in group.block_ids
        for bid in ids
    }
    assert sources and all(pool.blocks[bid].ref_cnt == 1 for bid in sources)
    scheduler.add_request(request_for("waiting", tokens=128, value=1))
    free = pool.get_num_free_blocks()
    start(scheduler, record)
    stored = {bid for job in offload._detached.values() for bid in job.source_ids}
    assert stored == sources
    assert pool.get_num_free_blocks() == free
    complete_jobs(scheduler)
    assert all(pool.blocks[bid].ref_cnt == 1 for bid in sources)
    assert peer in scheduler.running
    resumed = request_for("resumed", tokens=144)
    assert offload.get_num_new_matched_tokens(resumed, 0) == (128, True)


@pytest.mark.parametrize("constraint,expected", [("cpu", 8), ("duration", 10)])
def test_automatic_preservation_respects_cpu_and_restore_time(
    monkeypatch, constraint, expected
):
    scheduler = make_offload_scheduler(
        gpu_blocks=129,
        cpu_blocks=8 if constraint == "cpu" else 128,
        max_num_batched_tokens=1024,
    )
    record = complete_generation(scheduler, request_for("creator", tokens=640))
    offload = scheduler.connector.tokencake_scheduler
    monkeypatch.setattr(offload.lifecycles, "clock", lambda: 1.0)
    if constraint == "duration":
        monkeypatch.setattr(
            offload, "estimate_transfer", lambda blocks, direction: blocks * 0.1
        )
    scheduler.add_request(request_for("waiting", tokens=1024, value=1))
    offload.lifecycles.apply(
        LifecycleEvent(
            "stall_started", record.metadata.lifecycle_id, estimated_duration_s=6.1
        )
    )
    offload.evaluate_pending(scheduler)
    sources = [bid for job in offload._detached.values() for bid in job.source_ids]
    assert len(sources) == expected


@pytest.mark.parametrize(
    "cause",
    [
        "no_waiting",
        "low_pressure",
        "short_stall",
        "ineligible",
        "capacity",
        "threshold",
    ],
)
def test_policy_rejections(cause):
    scheduler = make_offload_scheduler()
    settings = scheduler.vllm_config._tokencake_config.offload
    metadata = TokenCakeMetadata(
        lifecycle_id=f"tc-{uuid4().hex}",
        offload_eligible=True,
        reusable_prefix=True,
    )
    pressure = Pressure(0.7, 32, 1, 64, 0, 32, 16)
    duration, available = 2.0, 64
    expected = Metric.SELECTED
    if cause == "no_waiting":
        pressure = replace(pressure, fit_demand=0)
        expected = Metric.NO_WAITING_DEMAND
    elif cause == "low_pressure":
        settings = replace(settings, min_gpu_usage=0.6)
        pressure = replace(pressure, usage=0.1, waiting_demand=16)
        expected = Metric.LOW_PRESSURE
    elif cause == "short_stall":
        duration = 0.001
        expected = Metric.UNPROFITABLE
    elif cause == "ineligible":
        metadata = metadata.model_copy(update={"offload_eligible": False})
        expected = Metric.NOT_ELIGIBLE
    elif cause == "capacity":
        available = 0
        expected = Metric.CPU_CAPACITY
    elif cause == "threshold":
        settings = replace(settings, score_threshold=100.0)
        expected = Metric.UNPROFITABLE
    result = evaluate_benefit(
        settings,
        metadata,
        pressure,
        blocks=16,
        cpu_available=available,
        cpu_needed=16,
        duration=duration,
        transfer_time=0.01,
    )
    assert result.reason == expected


def test_waiting_pressure_does_not_allocate_or_record_prefix_hits():
    scheduler = make_offload_scheduler()
    complete_generation(scheduler, request_for("creator"))
    scheduler.add_request(request_for("waiting", tokens=64))
    manager = scheduler.kv_cache_manager
    before = copy.deepcopy(manager.prefix_cache_stats)
    free = manager.block_pool.get_num_free_blocks()
    pressure = waiting_pressure(scheduler, 128, "first_fit")
    assert pressure.fit_demand > 0
    assert manager.prefix_cache_stats == before
    assert manager.block_pool.get_num_free_blocks() == free


@pytest.mark.parametrize("scheduling", [False, True])
@pytest.mark.parametrize("blocks,expected", [(64, 8), (17, 0)])
def test_preservation_window_bounds_observation_not_prefill_size(
    scheduling, blocks, expected
):
    scheduler = make_offload_scheduler(gpu_blocks=blocks, scheduling=scheduling)
    scheduler.add_request(request_for("waiting", tokens=512))
    controller = scheduler._tokencake_scheduling
    if controller is not None:
        controller.begin_step(scheduler.waiting, scheduler.running)
    manager = scheduler.kv_cache_manager
    free = manager.block_pool.get_num_free_blocks()
    before = scheduler._tokencake_lifecycles.metrics.snapshot(1)
    assert waiting_pressure(scheduler, 8, "first_fit").fit_demand == expected
    assert manager.block_pool.get_num_free_blocks() == free
    assert scheduler._tokencake_lifecycles.metrics.snapshot(1) == before


@pytest.mark.parametrize(
    "mode,available,fit", [("all", 3, 0), ("progress", 3, 0), ("reclaim", 7, 4)]
)
def test_waiting_pressure_respects_outstanding_generation_commitments(
    mode, available, fit
):
    scheduler = make_offload_scheduler(
        gpu_blocks=12, scheduling=True, generation_reserve_mode=mode
    )
    first = request_for("running", tokens=64)
    first.max_tokens = 64
    scheduler.add_request(first)
    scheduler.schedule()
    scheduler.add_request(request_for("waiting", tokens=64, value=1))
    controller = scheduler._tokencake_scheduling
    controller.begin_step(scheduler.waiting, scheduler.running)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 7
    assert controller.uncommitted_blocks == available
    assert waiting_pressure(scheduler, 128, "first_fit").fit_demand == fit


def test_waiting_pressure_accounts_for_borrowing_without_overcommitting():
    scheduler = make_offload_scheduler(scheduling=True)
    for name, tokens in (("first", 112), ("second", 192)):
        request = request_for(name, tokens=tokens)
        request.sampling_params.extra_args["tokencake"]["agent_type"] = name
        request.max_tokens = 64
        scheduler.add_request(request)
    controller = scheduler._tokencake_scheduling
    controller.begin_step(scheduler.waiting, scheduler.running)
    controller.plan = CapacityPlan(
        {"first", "second"},
        {"first": 10.0, "second": 9.0},
        {"first": 6, "second": 6},
        4,
    )
    controller.waiting_critical = {"first", "second"}
    controller.release()
    before = controller.metrics.snapshot(2)
    assert waiting_pressure(scheduler, 128, "first_fit").fit_demand == 7
    assert controller.metrics.snapshot(2) == before
    assert not controller.charges and not controller.reservation_deferred


def test_waiting_pressure_uses_decode_prefill_budget():
    scheduler = make_offload_scheduler(
        gpu_blocks=1025,
        scheduling=True,
        max_num_batched_tokens=8192,
        decode_prefill_token_budget=1024,
    )
    scheduler.add_request(request_for("decoder"))
    scheduler.schedule()
    scheduler.add_request(request_for("waiting", tokens=1536, value=1))
    controller = scheduler._tokencake_scheduling
    controller.begin_step(scheduler.waiting, scheduler.running)
    before = controller.metrics.snapshot(2)
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    assert waiting_pressure(scheduler, 128, "first_fit").fit_demand == 64
    assert controller.metrics.snapshot(2) == before
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free


def test_ordinary_store_progress_and_completion_delegate_to_native_connector():
    scheduler = make_offload_scheduler()
    ordinary = request_for("ordinary", annotated=False)
    annotated = request_for("annotated", value=1)
    scheduler.add_request(ordinary)
    scheduler.add_request(annotated)
    output = scheduler.schedule()
    offload = scheduler.connector.tokencake_scheduler
    jobs = output.kv_connector_metadata.store_jobs
    assert len(jobs) == 1
    assert next(iter(jobs.values())).req_id == ordinary.request_id
    assert (
        offload._req_status[ordinary.request_id].group_states[0].next_stored_block_idx
        == 2
    )
    assert (
        offload._req_status[annotated.request_id].group_states[0].next_stored_block_idx
        == 0
    )
    assert not offload._detached
    scheduler.finish_requests(ordinary.request_id, RequestStatus.FINISHED_STOPPED)
    keys = set().union(*(status.keys for status in offload._jobs.values()))
    offload.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata({j: 1 for j in jobs})
        )
    )
    assert ordinary.request_id not in offload._req_status
    assert all(offload.manager._policy.get(key).ref_cnt == 0 for key in keys)


def test_backoff_reacts_to_demand_and_preservation_does_not_add_gpu_capacity():
    scheduler = make_offload_scheduler()
    offload = scheduler.connector.tokencake_scheduler
    record = complete_generation(scheduler, request_for("creator"))
    start(scheduler, record)
    offload.evaluate_pending(scheduler, new_step=True)
    assert offload.lifecycles.metrics.snapshot(1)[Metric.BACKOFF.value] == 1
    scheduler.add_request(request_for("waiting", tokens=64, value=1))
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    offload.evaluate_pending(scheduler, new_step=True)
    assert offload.has_unpublished
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free
    first_jobs = set(offload._detached)
    offload.evaluate_pending(scheduler, new_step=True)
    assert set(offload._detached) == first_jobs


@pytest.mark.parametrize("cpu_ancestor", [False, True])
def test_snapshot_gap_requires_reachable_ancestor(cpu_ancestor):
    scheduler = make_offload_scheduler()
    offload = scheduler.connector.tokencake_scheduler
    record = complete_generation(scheduler, request_for("creator", tokens=128))
    group = record.snapshot.groups[0]
    if cpu_ancestor:
        offload.manager.prepare_store([group.keys[0]], record.snapshot.req_context)
        offload.manager.complete_store([group.keys[0]], record.snapshot.req_context)
    scheduler.kv_cache_manager.block_pool.evict_blocks(set(group.block_ids[0]))
    candidates = offload._snapshot_candidates(record)
    assert bool(candidates) == cpu_ancestor
    if cpu_ancestor:
        assert [c.key for c in candidates] == list(group.keys)
    else:
        counters = offload.lifecycles.metrics.snapshot(1)
        assert counters[Metric.STALE.value] == counters[Metric.PREFIX_GAP.value] == 1


@pytest.mark.parametrize("groups,expected", [(2, 3), ("hybrid", 0)])
def test_bounded_multi_group_selection_requires_common_native_hit(groups, expected):
    scheduler = make_offload_scheduler(gpu_blocks=65, groups=groups)
    offload = scheduler.connector.tokencake_scheduler
    record = complete_generation(scheduler, request_for("creator", tokens=128))
    candidates = offload._snapshot_candidates(record)
    assert len(offload._bound_candidates(candidates, 4)) == expected


@pytest.mark.parametrize("local_success", [False, True])
@pytest.mark.parametrize("connector_success", [False, True])
def test_reset_preserves_each_native_component_result(
    local_success, connector_success, monkeypatch
):
    scheduler = make_offload_scheduler()
    offload = scheduler.connector.tokencake_scheduler
    record = complete_generation(scheduler, request_for("creator"))
    waiter = request_for("waiting", tokens=64, value=1)
    scheduler.add_request(waiter)
    start(scheduler, record)
    complete_jobs(scheduler)
    keys = record.retained.copy()
    assert keys
    if not local_success:
        scheduler.schedule()
        assert scheduler.running
    if not connector_success:
        monkeypatch.setattr(scheduler.connector, "reset_cache", lambda: False)
    else:
        original = offload.manager.reset_cache

        def reset_native_first():
            assert record.retained == keys and record.terminal_at is None
            original()

        monkeypatch.setattr(offload.manager, "reset_cache", reset_native_first)
    assert scheduler.reset_prefix_cache(reset_connector=True) == (
        local_success and connector_success
    )
    if connector_success:
        assert record.terminal_cause == "reset" and not record.retained
        assert all(offload.manager._policy.get(key) is None for key in keys)
    else:
        assert record.terminal_at is None and record.retained == keys
        assert (record.snapshot is None) == local_success
        assert all(offload.manager._policy.get(key).ref_cnt == 1 for key in keys)


@pytest.mark.parametrize("published", [False, True])
def test_reset_only_flushes_jobs_known_to_the_worker(published):
    scheduler = make_offload_scheduler()
    offload = scheduler.connector.tokencake_scheduler
    record = complete_generation(scheduler, request_for("creator"))
    scheduler.add_request(request_for("waiting", tokens=64, value=1))
    start(scheduler, record)
    job_ids = set(offload._detached)
    worker, handler = worker_for(scheduler)
    if published:
        output = scheduler.schedule()
        worker.prepare_store_kv(output.kv_connector_metadata)
    assert scheduler.reset_connector_cache()
    output = scheduler.schedule()
    metadata = output.kv_connector_metadata
    assert metadata.jobs_to_flush == (job_ids if published else set())
    assert not metadata.store_jobs
    worker.handle_preemptions(metadata)
    assert handler.flushed_jobs == (job_ids if published else set())
    worker.get_finished(set())
    offload.update_connector_output(
        KVConnectorOutput(kv_connector_worker_meta=worker.build_connector_worker_meta())
    )
    assert not offload._jobs and not offload._detached and not record.retained


@pytest.mark.parametrize("failure", ["submission", "completion"])
@pytest.mark.parametrize("direction", ["d2h", "h2d"])
def test_native_worker_failure_after_ack_is_fail_closed(
    direction, failure, monkeypatch
):
    scheduler = make_offload_scheduler()
    offload = scheduler.connector.tokencake_scheduler
    record = complete_generation(scheduler, request_for("creator"))
    waiting = request_for("waiting", tokens=64, value=1)
    scheduler.add_request(waiting)
    assert start(scheduler, record).status_code == 200
    worker, handler = worker_for(scheduler)
    if direction == "d2h":
        publication = scheduler.schedule()
        worker.prepare_store_kv(publication.kv_connector_metadata)
        metadata = scheduler.schedule().kv_connector_metadata
        submit = worker.handle_preemptions
        job_ids = set(offload._detached)
    else:
        complete_jobs(scheduler)
        scheduler.finish_requests(waiting.request_id, RequestStatus.FINISHED_ABORTED)
        assert scheduler.reset_prefix_cache()
        scheduler.add_request(request_for("successor", tokens=48))
        metadata = scheduler.schedule().kv_connector_metadata
        job_ids = set(metadata.load_jobs)
        submit = worker.start_kv_transfers
    assert job_ids
    if failure == "submission":
        monkeypatch.setattr(handler, "transfer_async", lambda *args: False)
        with pytest.raises(AssertionError):
            submit(metadata)
    else:
        submit(metadata)
        handler.completed_transfers = [TransferResult(next(iter(job_ids)), False)]
        with pytest.raises(AssertionError):
            worker.get_finished(set())
    assert worker.build_connector_worker_meta() is None
    assert job_ids <= offload._jobs.keys()


@pytest.mark.parametrize(
    "mode,expected", [("first_fit", 2), ("best_fit", 8), ("priority_first", 8)]
)
@pytest.mark.parametrize("scheduling", [False, True])
def test_temporal_demand_uses_native_admission_and_requested_order(
    mode, expected, scheduling
):
    scheduler = make_offload_scheduler(scheduling=scheduling)
    scheduler.max_num_running_reqs = 1
    first = request_for("first", tokens=32)
    second = request_for("second", tokens=128, value=1)
    first.priority = 5
    second.priority = 0
    second.sampling_params.extra_args["tokencake"]["importance"] = 100.0
    scheduler.add_request(first)
    scheduler.add_request(second)
    controller = scheduler._tokencake_scheduling
    if controller is not None:
        controller.begin_step(scheduler.waiting, scheduler.running)
    before = scheduler._tokencake_lifecycles.metrics.snapshot(2)
    pressure = waiting_pressure(scheduler, 10, mode)
    assert pressure.fit_demand == expected
    assert list(scheduler.waiting) == [first, second]
    assert scheduler._tokencake_lifecycles.metrics.snapshot(2) == before


def test_transfer_ewma_and_metrics_survive_external_reset():
    scheduler = make_offload_scheduler()
    offload = scheduler.connector.tokencake_scheduler
    for size in [1000, 3000]:
        stats = OffloadingConnectorStats()
        stats.record_transfer(size, 0.01, ("GPU", "CPU"))
        offload.update_connector_output(KVConnectorOutput(kv_connector_stats=stats))
    assert offload._bandwidth["d2h"] == 200000
    before = offload.estimate_transfer(4, "d2h")
    metrics = offload.lifecycles.metrics.snapshot(0)
    assert scheduler.reset_connector_cache()
    assert offload.estimate_transfer(4, "d2h") == before
    assert offload.lifecycles.metrics.snapshot(0) == metrics


def test_store_completion_failure_keeps_native_fences_and_reports_failure(monkeypatch):
    scheduler = make_offload_scheduler()
    offload = scheduler.connector.tokencake_scheduler
    record = complete_generation(scheduler, request_for("creator"))
    scheduler.add_request(request_for("waiting", tokens=64, value=1))
    assert start(scheduler, record).status_code == 200
    scheduler.schedule()

    def fail(*args, **kwargs):
        raise RuntimeError("native store completion failed")

    monkeypatch.setattr(offload.manager, "complete_store", fail)
    with pytest.raises(RuntimeError, match="native store completion failed"):
        complete_jobs(scheduler)
    assert offload._detached and offload._block_id_to_pending_jobs
    assert not record.retained
    assert offload.lifecycles.metrics.snapshot(1)[Metric.TRANSFER_FAILURE.value] == 1
