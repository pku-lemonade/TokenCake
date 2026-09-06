# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline scheduler diagnosis; no model execution or timing claims.

Run from the repository root with HF_HUB_OFFLINE=1 and PYTHONPATH=.
The feedback tokens are synthetic. This is supporting mechanism evidence,
not a complete-DAG correctness or performance acceptance run.
"""

import copy
import json
from collections import Counter
from unittest.mock import patch

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.config.utils import replace
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.outputs import ModelRunnerOutput


def run(variant, agent_types):
    base = create_scheduler(
        model="/root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct",
        max_num_seqs=1024,
        max_num_batched_tokens=8192,
        max_model_len=32768,
        num_blocks=3050,
        enable_prefix_caching=True,
        async_scheduling=True,
    )
    if variant == "native":
        scheduler = base
    else:
        config = replace(
            base.vllm_config,
            additional_config={"tokencake": {"offload": {"enabled": False}}},
        )
        scheduler = AsyncScheduler(
            config,
            base.kv_cache_config,
            base.structured_output_manager,
            16,
            log_stats=True,
        )
    controller = scheduler._tokencake_scheduling
    if variant == "agent_without_prefill_cap":
        controller.prefill_limit = lambda request, tokens, *args, **kwargs: tokens
    if variant == "agent_with_native_victim":
        controller.victim = lambda request, running: None
    if variant == "agent_without_reservations":
        controller.can_allocate = lambda *args, **kwargs: True
        controller.commit = lambda *args, **kwargs: None
    requests = create_requests(24, num_tokens=8192, max_tokens=500, ignore_eos=True)
    for index, request in enumerate(requests):
        request.arrival_time = 1000.0 + index * 0.001
        if controller is not None:
            request.sampling_params = copy.deepcopy(request.sampling_params)
            request.sampling_params.extra_args = {
                "tokencake": {
                    "lifecycle_id": f"tc-{index:08x}000040008000000000000000",
                    "agent_type": f"agent-{index % agent_types}",
                    "importance": float(1 + index % agent_types),
                }
            }
        scheduler.add_request(request)
    events = []
    counters = Counter()
    if controller is not None:
        allocate = controller.can_allocate
        victim = controller.victim

        def trace_allocation(request, *args, **kwargs):
            allowed = allocate(request, *args, **kwargs)
            if not allowed:
                counters["reservation_denials"] += 1
                if request in scheduler.running:
                    counters["running_reservation_denials"] += 1
            return allowed

        def trace_victim(request, running):
            selected = victim(request, running)
            if selected is request:
                counters["self_preemptions"] += 1
            return selected

        controller.can_allocate = trace_allocation
        controller.victim = trace_victim
    previous_preemptions = 0
    first_step = None
    for step in range(20000):
        before = {
            r.request_id: (r.num_computed_tokens, r.num_output_tokens)
            for r in scheduler.running
        }
        with (
            patch("time.time", return_value=1000 + step * 0.02),
            patch("time.monotonic", return_value=5000 + step * 0.02),
        ):
            output = scheduler.schedule()
        if first_step is None:
            first_step = dict(output.num_scheduled_tokens)
        counters["scheduled_tokens"] += output.total_num_scheduled_tokens
        counters["steps"] += 1
        if not output.num_scheduled_tokens:
            counters["empty_steps"] += 1
        preemptions = sum(r.num_preemptions for r in requests)
        if preemptions > previous_preemptions:
            for request_id in output.preempted_req_ids or ():
                computed, generated = before[request_id]
                counters["computed_positions_before_preemption"] += computed
                if len(events) < 12:
                    events.append(
                        {
                            "step": step,
                            "request": request_id,
                            "computed": computed,
                            "generated": generated,
                            "running": len(scheduler.running),
                            "free_blocks": (
                                scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
                            ),
                        }
                    )
            previous_preemptions = preemptions
        ids = list(output.num_scheduled_tokens)
        sampled = [
            [] if scheduler.requests[key].is_prefill_chunk else [100] for key in ids
        ]
        with (
            patch("time.time", return_value=1000 + step * 0.02),
            patch("time.monotonic", return_value=5000 + step * 0.02),
        ):
            scheduler.update_from_output(
                output,
                ModelRunnerOutput(
                    req_ids=ids,
                    req_id_to_index={key: i for i, key in enumerate(ids)},
                    sampled_token_ids=sampled,
                    logprobs=None,
                    prompt_logprobs_dict={},
                    pooler_output=[],
                ),
            )
        if all(r.is_finished() for r in requests):
            break
    return {
        "variant": variant,
        "configuration": {
            "requests": 24,
            "agent_types": agent_types,
            "prompt_tokens_each": 8192,
            "max_output_tokens_each": 500,
            "num_gpu_blocks": 3050,
            "block_size": 16,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 1024,
            "scheduler_reserve_full_isl": scheduler.scheduler_reserve_full_isl,
            "prefix_caching": True,
            "async_scheduling": True,
            "synthetic_feedback": True,
            "virtual_seconds_per_step": 0.02,
        },
        "first_step": first_step,
        "completed": sum(r.is_finished() for r in requests),
        "generated_tokens": sum(r.num_output_tokens for r in requests),
        "preemptions": sum(r.num_preemptions for r in requests),
        "maximum_preemptions_per_request": max(r.num_preemptions for r in requests),
        "counters": dict(counters),
        "first_preemption_events": events,
    }


if __name__ == "__main__":
    for variant in (
        "native",
        "agent",
        "agent_without_prefill_cap",
        "agent_with_native_victim",
        "agent_without_reservations",
    ):
        print("DIAGNOSIS " + json.dumps(run(variant, 8), sort_keys=True), flush=True)
