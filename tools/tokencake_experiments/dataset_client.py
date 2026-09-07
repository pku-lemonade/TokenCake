# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute a JSON DAG, composing each successor from live predecessor outputs."""

import argparse
import asyncio
import json
import math
import time
import uuid
from graphlib import TopologicalSorter
from pathlib import Path

import aiohttp
import httpx

if __package__:
    from .dataset import (
        Application,
        Dataset,
        Node,
        compose_input,
        compose_output,
        load_dataset,
    )
else:
    from dataset import (
        Application,
        Dataset,
        Node,
        compose_input,
        compose_output,
        load_dataset,
    )

MODES = ("native", "agent", "offload", "offload-agent", "old-offload-agent")
REQUEST_TIMEOUT = 600000


def request_metadata(
    dataset: Dataset, node: Node, started: float, origin: float, identifier: str
) -> dict:
    metadata = node.metadata
    result = {
        "lifecycle_id": identifier,
        "agent_name": node.name,
        "agent_type": node.type,
        "importance": metadata.get("priority", 0),
        "application_started_at_s": started,
        "application_start_offset_s": started - origin,
        "application_elapsed_s": time.time() - started,
        "application_max_depth": dataset.max_depth,
        "remaining_depth": max(0, dataset.max_depth - metadata.get("depth", 0)),
    }
    for key in ("in_degree", "out_degree", "similarity", "depth"):
        result[key] = metadata.get(key, 0)
    for key in (
        "critical_path",
        "near_completion",
        "join_group",
        "dependency_depth",
        "fanout_width",
        "memory_weight",
        "offload_eligible",
        "reusable_prefix",
    ):
        if key in metadata:
            result[key] = metadata[key]
    return result


def legacy_metadata(
    dataset: Dataset, app: Application, node: Node, started: float, origin: float
) -> dict:
    fields = request_metadata(dataset, node, started, origin, "")
    names = {
        "agent_name": "name",
        "agent_type": "type",
        "importance": "priority",
        "application_started_at_s": "app_start_time",
        "application_start_offset_s": "app_start_offset",
        "application_elapsed_s": "app_elapsed_time",
        "application_max_depth": "app_max_depth",
    }
    result = {
        names.get(key, key): value
        for key, value in fields.items()
        if key != "lifecycle_id"
    }
    result.update(
        application_id=int(app.id) if app.id.isdecimal() else app.id,
        start_time=str(time.time()),
    )
    for key in (
        "workload_profile",
        "branch_id",
        "branch_group",
        "expected_tool_stall_s",
        "stage_type",
        "preserve_llm_output_after_tool",
    ):
        if key in node.metadata:
            result[key] = node.metadata[key]
    duration = result.get("expected_tool_stall_s", 0)
    if duration > 0:
        result["resume_deadline"] = time.time() + duration
    return result


async def post_event(
    base_url: str, event: dict, *, route: str = "tokencake/events"
) -> None:
    async with (
        aiohttp.ClientSession() as session,
        session.post(f"{base_url}/{route}", json=event) as response,
    ):
        response.raise_for_status()
        await response.read()


async def run_application(
    dataset: Dataset,
    app: Application,
    *,
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    mode: str,
    origin: float,
    arrival_offset: float = 0,
    sleep=asyncio.sleep,
) -> tuple[dict, dict]:
    started = time.time()
    if mode not in MODES:
        raise ValueError(f"Unknown client mode: {mode}")
    legacy = mode == "old-offload-agent"
    outputs: dict[str, list[str]] = {}
    records: dict[str, dict] = {}
    io_records: dict[str, dict] = {}
    nodes = {node.name: node for node in dataset.nodes}
    graph = TopologicalSorter(dataset.dependencies)
    graph.prepare()

    async def execute(node: Node) -> str:
        node_started = time.time()
        chunks = (
            [app.initial_input]
            if node.kind == "input"
            else compose_input(dataset, node, outputs)
        )
        llm_latency = tool_latency = 0.0
        identifier = None
        finish_reason = "local"
        usage = {"prompt_tokens": 0, "generated_tokens": 0, "processed_tokens": 0}
        if node.kind != "llm":
            outputs[node.name] = chunks
        else:
            prompt = "".join(chunks)
            llm_started = time.time()
            for attempt in range(4):
                identifier = f"tc-{uuid.uuid4().hex}"
                print(
                    "[TokenCakeAttempt] "
                    + json.dumps(
                        {
                            "application_id": app.id,
                            "node": node.name,
                            "attempt": attempt,
                            "request_id": identifier,
                            "prompt_chars": len(prompt),
                        }
                    ),
                    flush=True,
                )
                body = {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": node.max_tokens,
                    "ignore_eos": True,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "n": 1,
                    "request_id": identifier,
                }
                if legacy:
                    body["agent_info"] = legacy_metadata(
                        dataset, app, node, started, origin
                    )
                elif mode != "native":
                    body["vllm_xargs"] = {
                        "tokencake": request_metadata(
                            dataset, node, started, origin, identifier
                        )
                    }
                try:
                    response = await client.post(
                        f"{base_url}/completions", json=body, timeout=REQUEST_TIMEOUT
                    )
                    response.raise_for_status()
                    completion = response.json()
                    if legacy:
                        identifier = completion["id"]
                    generated = completion["choices"][0]["text"]
                    finish_reason = completion["choices"][0].get("finish_reason")
                    raw_usage = completion.get("usage") or {}
                    usage = {
                        "prompt_tokens": int(raw_usage.get("prompt_tokens", 0)),
                        "generated_tokens": int(raw_usage.get("completion_tokens", 0)),
                        "processed_tokens": int(
                            raw_usage.get("total_tokens", 0)
                            or (
                                raw_usage.get("prompt_tokens", 0)
                                + raw_usage.get("completion_tokens", 0)
                            )
                        ),
                    }
                    break
                except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
                    print(
                        f"[debug][{node.name}] {exc}, retrying... ({attempt + 1}/3)",
                        flush=True,
                    )
                    if attempt == 3:
                        raise RuntimeError(
                            "Failed to send LLM request after 3 retries"
                        ) from exc
                    prompt = prompt[: len(prompt) // 2]
            llm_latency = time.time() - llm_started
            io_records[node.name] = {
                "input": prompt,
                "output": generated,
                "request_id": identifier,
                "finish_reason": finish_reason,
                "latency": time.time() - node_started,
                **usage,
            }
            final_output = generated
            if node.tool is not None:
                tool_started = time.time()
                events = mode in ("offload", "offload-agent")
                if legacy:
                    await post_event(
                        base_url,
                        {
                            "request_id": identifier,
                            "estimated_time": node.tool.duration_s,
                            "request_type": node.tool.kind,
                        },
                        route="mcp",
                    )
                elif events:
                    event = {
                        "event": "stall_started",
                        "lifecycle_id": identifier,
                        "kind": node.tool.kind,
                    }
                    if node.tool.duration_s > 0:
                        event["estimated_duration_s"] = node.tool.duration_s
                    await post_event(base_url, event)
                await sleep(node.tool.duration_s)
                if legacy:
                    await post_event(
                        base_url,
                        {"request_id": identifier, "request_type": node.tool.kind},
                        route="mcp/finished",
                    )
                elif events:
                    await post_event(
                        base_url,
                        {"event": "stall_finished", "lifecycle_id": identifier},
                    )
                tool_latency = time.time() - tool_started
                result = app.tool_results.get(node.name, node.tool.result)
                final_output = compose_output(
                    generated, result, node.tool.preserve_generation
                )
            outputs[node.name] = chunks + [final_output]
        ended = time.time()
        duration = node.tool.duration_s if node.tool is not None else 0.0
        records[node.name] = {
            "name": node.name,
            "type": node.type,
            "execution_kind": "llm" if node.kind == "llm" else "local",
            "finish_reason": finish_reason,
            "request_id": identifier,
            "latency": ended - node_started,
            "llm_latency": llm_latency,
            "tool_latency": tool_latency,
            "residual_latency": max(
                0.0, ended - node_started - llm_latency - tool_latency
            ),
            "base_tool_latency": duration,
            "scheduled_tool_latency": duration,
            "estimated_tool_latency": duration,
            "sampled_actual_tool_latency": duration,
            "tool_prediction_error_s": 0.0,
            "is_mcp": node.tool is not None,
            "start_time": node_started - origin,
            "end_time": ended - origin,
            **usage,
        }
        return node.name

    pending: set[asyncio.Task] = set()
    tasks: list[asyncio.Task] = []
    try:
        while graph.is_active():
            for name in graph.get_ready():
                task = asyncio.create_task(execute(nodes[name]))
                tasks.append(task)
                pending.add(task)
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                graph.done(task.result())
    finally:
        for task in pending:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    ended = time.time()
    return {
        "app_start_time": started - origin,
        "app_end_time": ended - origin,
        "app_latency": ended - started,
        "arrival_offset_s": arrival_offset,
        "app_finished": len(records) == len(nodes),
        "application_internal_finished": not graph.is_active(),
        "expected_node_names": sorted(nodes),
        "expected_node_names_unique": True,
        "observed_node_names": sorted(records),
        "request_info": records,
        "workload_profile": dataset.name,
        "context_token_count": app.context_token_count,
        "context_sources": app.context_sources,
        "context_source_revision": app.context_source_revision,
        "frozen_workload_contract": dataset.contract(app),
    }, io_records


async def benchmark(dataset: Dataset, args: argparse.Namespace) -> None:
    if args.num_requests is not None and args.num_requests != len(dataset.applications):
        raise ValueError("num_requests must match the complete JSON dataset")
    if not math.isfinite(args.request_rate) or args.request_rate <= 0:
        raise ValueError("request_rate must be positive and finite")
    offsets = [index / args.request_rate for index in range(len(dataset.applications))]
    if args.arrival_trace_file is not None:
        payload = json.loads(args.arrival_trace_file.read_text())
        offsets = payload["offsets_s"]
    if (
        len(offsets) != len(dataset.applications)
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
            for value in offsets
        )
        or offsets != sorted(offsets)
    ):
        raise ValueError("Invalid application arrival trace")
    origin = time.time()
    async with httpx.AsyncClient() as client:

        async def scheduled(app: Application, offset: float):
            await asyncio.sleep(max(0.0, origin + offset - time.time()))
            return await run_application(
                dataset,
                app,
                client=client,
                base_url=f"http://127.0.0.1:{args.port}/v1",
                model=args.model_path,
                mode=args.tokencake_mode,
                origin=origin,
                arrival_offset=offset,
            )

        tasks = [
            asyncio.create_task(scheduled(app, offset))
            for app, offset in zip(dataset.applications, offsets)
        ]
        try:
            results = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apps = {str(index): result[0] for index, result in enumerate(results)}
    with (
        args.output_dir / f"app_qps_{args.request_rate}_num_{len(results)}.json"
    ).open("x") as stream:
        json.dump(apps, stream)
    with args.output_file.open("x") as stream:
        json.dump(
            {str(index): result[1] for index, result in enumerate(results)}, stream
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--request_rate", type=float, required=True)
    parser.add_argument("--num_requests", type=int)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--arrival_trace_file", type=Path)
    parser.add_argument("--tokencake-mode", choices=MODES, default="native")
    parser.add_argument("--disable_mcp_notifications", action="store_true")
    parser.add_argument("--task", default="code-paper-pressure")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workload_source_revision", default="")
    args = parser.parse_args()
    dataset = load_dataset(args.dataset)
    if args.task != dataset.name or args.seed != dataset.provenance["seed"]:
        raise ValueError("Client arguments differ from the fixed JSON dataset")
    if args.disable_mcp_notifications and args.tokencake_mode in (
        "offload",
        "offload-agent",
        "old-offload-agent",
    ):
        raise ValueError("Tool events are required by the selected offload mode")
    asyncio.run(benchmark(dataset, args))


if __name__ == "__main__":
    main()
