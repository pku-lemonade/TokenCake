# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import FrozenInstanceError

import pytest
import torch
from transformers import OPTConfig

from vllm.config import (
    CacheConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
)
from vllm.config.utils import replace
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.tokencake.config import parse_tokencake_config
from vllm.tokencake.offloading import TokenCakeConnector
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec


def test_absent_and_immutable_defaults(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_AGENT_SCHEDULING", "1")
    monkeypatch.setenv("VLLM_TEMPORAL_SELECTION_POLICY", "best_fit")
    assert parse_tokencake_config({}) is None
    settings = parse_tokencake_config({"tokencake": {}})
    assert settings is not None
    assert settings.scheduling.enabled and settings.offload.enabled
    assert settings.temporal_selection == "first_fit"
    assert settings.scheduling.reserve_ratio_min == 0.05
    assert settings.scheduling.reserve_generation_tokens
    assert settings.scheduling.decode_prefill_token_budget == 0
    assert settings.offload.transfer.d2h_bandwidth_gbps == 12.0
    with pytest.raises(FrozenInstanceError):
        settings.scheduling.enabled = False
    with pytest.raises(FrozenInstanceError):
        settings.offload.transfer.d2h_bandwidth_gbps = 1.0


@pytest.mark.parametrize("scheduling", [False, True])
@pytest.mark.parametrize("offload", [False, True])
def test_independent_switches(scheduling, offload):
    settings = parse_tokencake_config(
        {
            "tokencake": {
                "scheduling": {"enabled": scheduling},
                "offload": {"enabled": offload},
            }
        }
    )
    assert settings is not None
    assert settings.scheduling.enabled == scheduling
    assert settings.offload.enabled == offload
    assert settings.temporal_selection == "first_fit"


@pytest.mark.parametrize("policy", ["first_fit", "best_fit", "priority_first"])
def test_temporal_selection(policy):
    settings = parse_tokencake_config(
        {"tokencake": {"scheduling": {"temporal_selection": policy}}}
    )
    assert settings is not None and settings.temporal_selection == policy


@pytest.mark.parametrize(
    "value",
    [
        None,
        False,
        [],
        "enabled",
        {"unknown": {}},
        {"scheduling": {"unknown": 1}},
        {"offload": {"unknown": 1}},
        {"offload": {"transfer": {"unknown": 1}}},
        {"scheduling": {"enabled": 1}},
        {"scheduling": {"reserve_generation_tokens": 1}},
        {"scheduling": {"decode_prefill_token_budget": -1}},
        {"scheduling": {"decode_prefill_token_budget": True}},
        {"scheduling": {"cache_affinity_score_band": -1}},
        {"scheduling": {"cache_affinity_score_band": True}},
        {"offload": {"enabled": "true"}},
        {"scheduling": {"enabled": False, "critical_ratio": 0.75}},
        {"offload": {"enabled": False, "transfer": {}}},
        {"scheduling": {"temporal_selection": "fifo"}},
        {"scheduling": {"critical_ratio": 0}},
        {"scheduling": {"reserve_ratio_min": 0.4, "reserve_ratio_max": 0.3}},
        {"scheduling": {"gpu_usage_low": 0.75}},
        {"scheduling": {"reserve_adjustment_step": True}},
        {"scheduling": {"reserve_ratio_max": float("nan")}},
        {"scheduling": {"gpu_usage_high": 1.1}},
        {"offload": {"min_gpu_usage": 0.9}},
        {"offload": {"score_threshold": -1.0}},
        {"offload": {"backoff_steps": -1}},
        {"offload": {"backoff_steps": 1.5}},
        {"offload": {"backoff_steps": True}},
        {"offload": {"eviction_window_blocks": 0}},
        {"offload": {"max_relief_blocks": 0}},
        {"offload": {"default_stall_s": 0}},
        {"offload": {"release_lead_s": -0.1}},
        {"offload": {"ewma_alpha": 0}},
        {"offload": {"transfer": {"d2h_bandwidth_gbps": 0}}},
        {"offload": {"transfer": {"h2d_bandwidth_gbps": "12"}}},
        {"offload": {"transfer": {"d2h_base_time_s": float("inf")}}},
    ],
)
def test_reject_invalid_config(value):
    with pytest.raises((ValueError, TypeError)):
        parse_tokencake_config({"tokencake": value})


@pytest.fixture(autouse=True)
def native_connector(monkeypatch):
    monkeypatch.setenv("VLLM_USE_SIMPLE_KV_OFFLOAD", "0")


@pytest.fixture(scope="module")
def model_config(tmp_path_factory):
    path = tmp_path_factory.mktemp("tokencake-model")
    OPTConfig(architectures=["OPTForCausalLM"]).save_pretrained(path)
    return ModelConfig(model=str(path), skip_tokenizer_init=True)


@pytest.mark.parametrize(
    "settings,capacity,connector",
    [
        (None, None, None),
        (None, 1.0, "OffloadingConnector"),
        ({"scheduling": {"enabled": False}, "offload": {"enabled": False}}, None, None),
        (
            {"scheduling": {"enabled": False}, "offload": {"enabled": False}},
            1.0,
            "OffloadingConnector",
        ),
        ({"offload": {"enabled": False}}, None, None),
        ({"offload": {"enabled": False}}, 1.0, "OffloadingConnector"),
        ({"scheduling": {"enabled": False}}, 1.0, "TokenCakeConnector"),
        ({}, 1.0, "TokenCakeConnector"),
    ],
)
def test_startup_connector_selection(settings, capacity, connector, model_config):
    config = VllmConfig(
        model_config=model_config,
        additional_config={} if settings is None else {"tokencake": settings},
        cache_config=CacheConfig(kv_offloading_size=capacity),
    )
    actual = (
        config.kv_transfer_config.kv_connector if config.kv_transfer_config else None
    )
    assert actual == connector
    if settings is None:
        assert config._tokencake_config is None
    # Derived immutable settings must also survive the native config-copy path.
    assert replace(config)._tokencake_config == config._tokencake_config


@pytest.mark.parametrize("capacity", [None, 0.0, -1.0, float("inf")])
def test_offload_capacity_required(capacity):
    with pytest.raises(ValueError, match="kv_offloading_size"):
        VllmConfig(
            additional_config={"tokencake": {}},
            cache_config=CacheConfig(kv_offloading_size=capacity),
        )


@pytest.fixture
def kv_cache_config():
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.float16
    )
    return KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[
            KVCacheTensor(size=16 * spec.page_size_bytes, shared_by=["layer"])
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer"], kv_cache_spec=spec)],
    )


def test_native_cpu_manager_construction(model_config, kv_cache_config, monkeypatch):
    config = VllmConfig(
        model_config=model_config,
        additional_config={"tokencake": {}},
        cache_config=CacheConfig(kv_offloading_size=0.001),
    )
    connector = KVConnectorFactory.create_connector(
        config, KVConnectorRole.SCHEDULER, kv_cache_config
    )
    assert isinstance(connector, TokenCakeConnector)
    assert isinstance(connector.connector_scheduler.manager, CPUOffloadingManager)
    monkeypatch.setattr(CPUOffloadingSpec, "get_manager", lambda self: object())
    with pytest.raises(ValueError, match="concrete CPUOffloadingManager"):
        TokenCakeConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)


def test_capacity_must_fit_a_block(model_config, kv_cache_config):
    config = VllmConfig(
        model_config=model_config,
        additional_config={"tokencake": {}},
        cache_config=CacheConfig(kv_offloading_size=1 / 2**30),
    )
    with pytest.raises(ValueError, match="at least one KV block"):
        TokenCakeConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)
    config.additional_config = {}
    config._post_init_kv_transfer_config()
    with pytest.raises(ValueError, match="structured TokenCake offload enablement"):
        TokenCakeConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)


def test_backend_and_connector_mismatch():
    with pytest.raises(ValueError, match="backend"):
        VllmConfig(
            additional_config={"tokencake": {}},
            cache_config=CacheConfig(
                kv_offloading_size=1, kv_offloading_backend="lmcache"
            ),
        )
    with pytest.raises(ValueError, match="incompatible"):
        VllmConfig(
            additional_config={"tokencake": {}},
            cache_config=CacheConfig(kv_offloading_size=1),
            kv_transfer_config=KVTransferConfig(
                kv_connector="ExampleConnector", kv_role="kv_both"
            ),
        )
    with pytest.raises(ValueError, match="CPUOffloadingSpec"):
        VllmConfig(
            additional_config={"tokencake": {}},
            cache_config=CacheConfig(kv_offloading_size=1),
            kv_transfer_config=KVTransferConfig(
                kv_connector="OffloadingConnector",
                kv_role="kv_both",
                kv_connector_extra_config={"spec_name": "TieringOffloadingSpec"},
            ),
        )


def test_simple_fallback_rejected_only_for_tokencake_offload(monkeypatch):
    monkeypatch.setenv("VLLM_USE_SIMPLE_KV_OFFLOAD", "1")
    with pytest.raises(ValueError, match="VLLM_USE_SIMPLE_KV_OFFLOAD=0"):
        VllmConfig(
            additional_config={"tokencake": {}},
            cache_config=CacheConfig(kv_offloading_size=1),
        )
    native = VllmConfig(cache_config=CacheConfig(kv_offloading_size=1))
    assert native.kv_transfer_config.kv_connector == "SimpleCPUOffloadConnector"


@pytest.mark.parametrize(
    "scheduling,offload", [(True, False), (False, True), (True, True)]
)
def test_data_parallel_rejected(scheduling, offload):
    with pytest.raises(ValueError, match="data_parallel_size=1"):
        VllmConfig(
            parallel_config=ParallelConfig(data_parallel_size=2),
            cache_config=CacheConfig(kv_offloading_size=1 if offload else None),
            additional_config={
                "tokencake": {
                    "scheduling": {"enabled": scheduling},
                    "offload": {"enabled": offload},
                }
            },
        )
