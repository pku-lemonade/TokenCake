# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import weakref
from uuid import uuid4

import pytest

from vllm.tokencake.config import OffloadConfig
from vllm.tokencake.events import LifecycleEvent
from vllm.tokencake.lifecycle import (
    DuplicateLifecycleError,
    LifecycleRegistry,
    PrefixSnapshot,
    SnapshotGroup,
)
from vllm.tokencake.protocol import TokenCakeMetadata
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def registry():
    return LifecycleRegistry(OffloadConfig(), CPUOffloadingManager(8), clock=Clock())


def associate(registry, **metadata):
    identifier = f"tc-{uuid4().hex}"
    registry.associate(
        f"native-{identifier}", TokenCakeMetadata(lifecycle_id=identifier, **metadata)
    )
    return identifier


def start(identifier, **kwargs):
    return LifecycleEvent("stall_started", identifier, **kwargs)


def finish(identifier):
    return LifecycleEvent("stall_finished", identifier)


def ready_key(registry, value):
    key = make_offload_key(str(value).encode(), 0)
    context = ReqContext("native")
    assert registry.manager.prepare_store([key], context)
    registry.manager.complete_store([key], context)
    return key


@pytest.mark.parametrize("completion_first", [False, True])
def test_completion_and_start_are_independent(registry, completion_first):
    identifier = associate(registry)
    snapshot = PrefixSnapshot((), ReqContext("native"))
    if completion_first:
        registry.generation_finished(identifier, "completed", snapshot)
        assert registry.records[identifier].state == "awaiting_start"
    else:
        assert registry.apply(start(identifier)).disposition == "applied"
        assert registry.records[identifier].snapshot is None
    if completion_first:
        registry.apply(start(identifier))
    else:
        registry.generation_finished(identifier, "completed", snapshot)
    record = registry.records[identifier]
    assert record.snapshot is snapshot and record.pending_evaluation
    assert record.state == "active"


@pytest.mark.parametrize("cause", ["completed", "aborted", "error"])
def test_first_terminal_transition_wins(registry, cause):
    identifier = associate(registry, agent_type="agent")
    event = start(identifier, estimated_duration_s=10)
    registry.apply(event)
    registry.clock.now = 2
    assert registry.apply(finish(identifier)).disposition == "applied"
    record = registry.records[identifier]
    original = record.__dict__.copy()
    history = dict(registry._history)
    registry.clock.now = 3
    registry.generation_finished(
        identifier, cause, PrefixSnapshot((), ReqContext("late"))
    )
    registry.external_reset_succeeded()
    assert registry.apply(event).disposition == "duplicate"
    assert registry.apply(finish(identifier)).disposition == "duplicate"
    assert record.__dict__ == original
    assert dict(registry._history) == history


@pytest.mark.parametrize("started", [False, True])
@pytest.mark.parametrize("cause", ["aborted", "error"])
def test_abort_and_error_do_not_train(registry, started, cause):
    identifier = associate(registry, agent_type="agent")
    if started:
        registry.apply(start(identifier))
    registry.clock.now = 0.5
    registry.generation_finished(identifier, cause)
    record = registry.records[identifier]
    assert record.snapshot is None and record.terminal_cause == cause
    assert not registry._history
    assert registry.apply(finish(identifier)).disposition == "late_finish"
    registry.clock.now = 60.5
    assert registry.apply(finish(identifier)).status_code == 404


def test_duplicate_association_and_retry_isolation(registry):
    identifier = associate(registry)
    record = registry.records[identifier]
    with pytest.raises(DuplicateLifecycleError):
        registry.associate("different-generation", record.metadata)
    assert registry.records[identifier] is record
    retry = associate(registry)
    assert registry.apply(start(retry)).status_code == 200
    assert record.started_at is None
    registry.apply(start(identifier))
    registry.apply(finish(identifier))
    registry.clock.now = 60
    registry.expire()
    assert identifier not in registry.records
    # A still-running generation cannot lose exclusivity to tombstone expiry.
    with pytest.raises(DuplicateLifecycleError):
        registry.associate("different-generation", record.metadata)
    registry.generation_finished(identifier, "completed")
    assert identifier not in registry._generations


def test_attach_and_tombstone_boundaries(registry):
    identifier = associate(registry)
    registry.generation_finished(identifier, "completed")
    registry.clock.now = 59.999
    assert registry.records[identifier].state == "awaiting_start"
    registry.clock.now = 60
    assert registry.apply(start(identifier)).status_code == 409
    record = registry.records[identifier]
    assert record.terminal_at == 60
    registry.clock.now = 119.999
    assert registry.apply(finish(identifier)).disposition == "late_finish"
    assert record.terminal_at == 60
    registry.clock.now = 120
    assert registry.apply(finish(identifier)).status_code == 404


@pytest.mark.parametrize("estimate,deadline", [(1, 60), (20, 80), (2000, 3600)])
def test_active_safety_bounds(registry, estimate, deadline):
    identifier = associate(registry, agent_type="agent")
    registry.apply(start(identifier, estimated_duration_s=estimate))
    registry.clock.now = deadline - 0.001
    registry.expire()
    assert registry.records[identifier].terminal_at is None
    registry.clock.now = deadline
    registry.expire()
    assert registry.records[identifier].terminal_cause == "expired"
    assert not registry._history


def test_prediction_four_cases_and_history_lru(registry):
    identifier = associate(registry, agent_type="agent")
    registry.apply(start(identifier))
    assert registry.records[identifier].predicted_duration == 1
    registry.clock.now = 0.4
    registry.apply(finish(identifier))
    next_id = associate(registry, agent_type="agent")
    registry.apply(start(next_id))
    assert registry.records[next_id].predicted_duration == 0.4
    blend_id = associate(registry, agent_type="agent")
    registry.apply(start(blend_id, estimated_duration_s=2))
    assert registry.records[blend_id].predicted_duration == 1.2
    estimate_id = associate(registry, agent_name="different")
    registry.apply(start(estimate_id, estimated_duration_s=2))
    assert registry.records[estimate_id].predicted_duration == 2
    for index in range(4096):
        identifier = associate(registry, agent_type=f"type-{index}")
        registry.apply(start(identifier))
        registry.apply(finish(identifier))
    assert len(registry._history) == 4096
    assert ("type", "agent", "stall") not in registry._history
    registry.clock.now += 60
    registry.expire()
    assert len(registry._history) == 4096


def test_anonymous_history_dies_with_tombstone(registry):
    identifier = associate(registry)
    registry.apply(start(identifier))
    registry.clock.now = 0.5
    registry.apply(finish(identifier))
    record_ref = weakref.ref(registry.records[identifier])
    assert registry.records[identifier].anonymous_sample == 0.5
    assert not registry._history
    registry.generation_finished(identifier, "completed")
    registry.clock.now = 60.5
    registry.expire()
    gc.collect()
    assert record_ref() is None


def test_replay_and_conflict_have_no_duration_side_effect(registry):
    identifier = associate(registry, agent_type="agent")
    assert registry.apply(finish(identifier)).status_code == 409
    event = start(identifier, estimated_duration_s=10)
    registry.apply(event)
    registry.clock.now = 2
    assert registry.apply(event).disposition == "duplicate"
    assert registry.apply(start(identifier, estimated_duration_s=20)).status_code == 409
    assert registry.records[identifier].started_at == 0
    assert not registry._history
    registry.apply(finish(identifier))
    assert registry._history["type", "agent", "stall"] == 2
    registry.clock.now = 3
    assert registry.apply(event).disposition == "duplicate"
    assert registry.apply(start(identifier)).status_code == 409
    assert registry.apply(finish(identifier)).disposition == "duplicate"
    assert registry._history["type", "agent", "stall"] == 2


def test_predicted_release_stacks_with_native_load_and_other_lifecycle(registry):
    key = ready_key(registry, 1)
    ids = [associate(registry), associate(registry)]
    for identifier in ids:
        registry.apply(start(identifier, estimated_duration_s=5))
        registry.retain_ready(registry.records[identifier], [key, key])
        assert not registry.retain_ready(registry.records[identifier], [key])
    registry.manager.prepare_load([key], ReqContext("load"))
    block = registry.manager._policy.get(key)
    assert block.ref_cnt == 3
    record = registry.records[ids[0]]
    registry.set_h2d_estimate(record, 1)
    assert record.release_deadline == 3.75
    registry.clock.now = 3.75
    registry.expire()
    assert block.ref_cnt == 2 and not record.retained
    registry.apply(finish(ids[0]))
    assert block.ref_cnt == 2
    registry.apply(finish(ids[1]))
    assert block.ref_cnt == 1
    registry.manager.complete_load([key], ReqContext("load"))
    assert block.ref_cnt == 0 and registry.manager.lookup(key, ReqContext("lookup"))


@pytest.mark.parametrize("cause", ["finish", "expiry", "reset"])
def test_inflight_store_completion_cannot_restore_released_ownership(registry, cause):
    identifier = associate(registry)
    registry.apply(start(identifier))
    record = registry.records[identifier]
    key = make_offload_key(b"key", 0)
    context = ReqContext("store")
    registry.manager.prepare_store([key], context)
    assert not registry.retain_ready(record, [key])
    assert registry.manager._policy.get(key).ref_cnt == -1
    if cause == "finish":
        registry.apply(finish(identifier))
    elif cause == "expiry":
        registry.clock.now = 0.9
        registry.expire()
    else:
        registry.manager.reset_cache()
        registry.external_reset_succeeded()
        registry.manager.prepare_store([key], context)
    registry.manager.complete_store([key], context)
    assert not registry.retain_ready(record, [key])
    assert registry.manager._policy.get(key).ref_cnt == 0


def test_local_reset_drops_only_snapshot_and_external_reset_preserves_history(registry):
    identifier = associate(registry, agent_type="agent")
    registry.generation_finished(
        identifier, "completed", PrefixSnapshot((), ReqContext("native"))
    )
    registry.apply(start(identifier, estimated_duration_s=10))
    record = registry.records[identifier]
    key = ready_key(registry, 1)
    registry.retain_ready(record, [key])
    registry.invalidate_snapshots()
    assert record.snapshot is None and record.retained == {key}
    assert record.started_at == 0
    registry.clock.now = 2
    registry.apply(finish(identifier))
    history = dict(registry._history)
    registry.manager.reset_cache()
    registry.external_reset_succeeded()
    assert dict(registry._history) == history
    assert record.terminal_cause == "finished"


def test_snapshot_revalidation_and_no_block_pinning():
    pool = BlockPool(4, True, 16)
    block = pool.get_new_blocks(1)[0]
    original = make_block_hash_with_group_id(BlockHash(b"hash"), 0)
    block.block_hash = original
    snapshot = SnapshotGroup(
        (make_offload_key(b"hash", 0),), ((block.block_id,),), ((original,),)
    )
    assert not snapshot.is_valid(0, pool)
    pool.free_blocks([block])
    assert block.ref_cnt == 0 and snapshot.is_valid(0, pool)
    block.reset_hash()
    block.block_hash = make_block_hash_with_group_id(BlockHash(b"different"), 0)
    assert not snapshot.is_valid(0, pool)


@pytest.mark.parametrize("policy", ["lru", "arc"])
def test_ready_retention_and_eviction(policy):
    manager = CPUOffloadingManager(1, cache_policy=policy)
    context = ReqContext("native")
    first, second = make_offload_key(b"first", 0), make_offload_key(b"second", 0)
    assert manager.retain([first]) == set()
    manager.prepare_store([first], context)
    assert manager.retain([first]) == set()
    assert manager._policy.get(first).ref_cnt == -1
    manager.complete_store([first], context)
    assert manager.retain([first, first]) == {first}
    assert manager.prepare_store([second], context) is None
    manager.release({first})
    assert manager.lookup(first, context)
    assert manager.prepare_store([second], context) is not None
    with pytest.raises(AssertionError):
        manager.release({second})
