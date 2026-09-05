# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real CUDA copies through TokenCake scheduling and native CPU offload."""

import pytest
import torch

from tests.tokencake.test_offloading import (
    complete_generation,
    make_offload_scheduler,
    request_for,
    start,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus


def collect_output(worker):
    _, received = worker.get_finished(set())
    return KVConnectorOutput(
        finished_recving=received,
        kv_connector_worker_meta=worker.build_connector_worker_meta(),
        kv_connector_stats=worker.get_kv_connector_stats(),
    )


@pytest.mark.parametrize("groups,factor", [(1, 1), (1, 2), (2, 1), ("hybrid", 1)])
@torch.inference_mode()
def test_preserved_data_survives_source_overwrite_and_native_prefix_reload(
    groups, factor, monkeypatch, default_vllm_config
):
    monkeypatch.setenv("VLLM_USE_SIMPLE_KV_OFFLOAD", "0")
    scheduler = make_offload_scheduler(
        gpu_blocks={1: 17, 2: 25, "hybrid": 33}[groups], groups=groups, factor=factor
    )
    offload = scheduler.connector.tokencake_scheduler
    spec = CPUOffloadingSpec(scheduler.vllm_config, scheduler.kv_cache_config)
    tensors = []
    refs = []
    for index, group in enumerate(scheduler.kv_cache_config.kv_cache_groups):
        page_size = group.kv_cache_spec.page_size_bytes
        data = (
            torch.arange(
                scheduler.kv_cache_config.num_blocks * page_size,
                device="cuda",
                dtype=torch.int64,
            )
            .add(index * 17)
            .remainder(127)
            .to(torch.int8)
            .reshape(-1, page_size)
        )
        tensors.append(CanonicalKVCacheTensor(data, page_size))
        refs.append([CanonicalKVCacheRef(index, page_size)])
    worker = OffloadingConnectorWorker(spec)
    worker._register_handlers(CanonicalKVCaches(tensors, refs))
    try:
        original = request_for("creator", tokens=128)
        record = complete_generation(scheduler, original)
        snapshot = record.snapshot
        assert snapshot is not None
        expected = {
            (group_index, block_index): tensors[group_index].tensor[list(ids)].clone()
            for group_index, group in enumerate(snapshot.groups)
            for block_index, ids in enumerate(group.block_ids)
            if all(ids)
        }
        waiter = request_for("overwrite", tokens=256, value=1)
        scheduler.add_request(waiter)
        assert start(scheduler, record).status_code == 200
        publication = scheduler.schedule()
        metadata = publication.kv_connector_metadata
        assert metadata.store_jobs and publication.total_num_scheduled_tokens == 0
        worker.handle_preemptions(metadata)
        worker.start_kv_transfers(metadata)
        worker.prepare_store_kv(metadata)
        assert worker.build_connector_worker_meta() is None
        assert record.pending_retention and not record.retained

        following = scheduler.schedule()
        assert following.num_scheduled_tokens[waiter.request_id] > 0
        next_meta = following.kv_connector_metadata
        assert metadata.store_jobs.keys() <= next_meta.jobs_to_flush
        worker.handle_preemptions(next_meta)
        worker.start_kv_transfers(next_meta)
        # Model writes obey exactly the scheduler's native source-reuse fences.
        allocated = scheduler.kv_cache_manager.get_blocks(waiter.request_id).blocks
        for group_index, blocks in enumerate(allocated):
            ids = [b.block_id for b in blocks if b.block_id]
            tensors[group_index].tensor[ids] = -42
        worker.worker.wait(set(metadata.store_jobs))
        output = collect_output(worker)
        assert output.kv_connector_stats.data["GPU_to_CPU"]
        offload.update_connector_output(output)
        assert record.retained and not record.pending_retention
        assert not offload._block_id_to_pending_jobs
        scheduler.finish_requests(waiter.request_id, RequestStatus.FINISHED_STOPPED)

        # Drop local hashes and bytes so the successor must use CPU-resident KV.
        assert scheduler.reset_prefix_cache(reset_connector=False)
        for tensor in tensors:
            tensor.tensor.fill_(-99)
        successor = request_for("successor", tokens=144)
        scheduler.add_request(successor)
        loading = scheduler.schedule()
        load_meta = loading.kv_connector_metadata
        assert load_meta.load_jobs
        assert successor.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        assert set(load_meta.load_jobs).isdisjoint(metadata.store_jobs)
        worker.handle_preemptions(load_meta)
        worker.start_kv_transfers(load_meta)
        worker.worker.wait(set(load_meta.load_jobs))
        output = collect_output(worker)
        assert output.finished_recving == {successor.request_id}
        assert output.kv_connector_stats.data["CPU_to_GPU"]
        offload.update_connector_output(output)
        allocated = scheduler.kv_cache_manager.get_blocks(successor.request_id).blocks
        verified = 0
        for group_index, group in enumerate(snapshot.groups):
            for index, key in enumerate(group.keys):
                if key not in record.retained or (group_index, index) not in expected:
                    continue
                targets = allocated[group_index][index * factor : (index + 1) * factor]
                if not targets or any(b.block_id == 0 for b in targets):
                    continue
                actual = tensors[group_index].tensor[[b.block_id for b in targets]]
                torch.testing.assert_close(actual, expected[group_index, index])
                verified += 1
        assert verified == len(record.retained) > 0
        assert offload._bandwidth.keys() == {"d2h", "h2d"}
    finally:
        worker.shutdown()
