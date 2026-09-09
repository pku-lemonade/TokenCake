# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare two independent GPU services under one shared wall-clock budget."""

import fcntl
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager

from .budget import Budget
from .experiment import preflight_environments, run_tasks
from .inputs import digest
from .service import Service
from .transport import write_json


@contextmanager
def paired_services(platform, output, modes, gpu_by_mode):
    cancelled = threading.Event()
    entered = []

    def start(mode):
        gpu = gpu_by_mode[mode]
        server = Service(
            platform,
            output / mode / "service",
            mode,
            gpu=gpu,
            port=8060 + 10 * gpu,
            cancel_event=cancelled,
        )
        try:
            server.__enter__()
        except BaseException:
            cancelled.set()
            raise
        entered.append(server)
        return server

    try:
        with ThreadPoolExecutor(max_workers=len(modes)) as pool:
            futures = {pool.submit(start, mode): mode for mode in modes}
            try:
                servers = {
                    futures[future]: future.result() for future in as_completed(futures)
                }
            except BaseException:
                cancelled.set()
                raise
        yield servers
    finally:
        cancelled.set()
        with ExitStack() as cleanup:
            for server in entered:
                cleanup.callback(server.__exit__, None, None, None)


def parallel_task_runs(
    platform,
    manifest,
    benchmark,
    tasks,
    modes,
    output,
    servers,
    environment_root,
    budget_s,
):
    stop = threading.Event()
    barrier = threading.Barrier(len(modes))
    summaries = {}

    def run_mode(mode):
        try:
            barrier.wait(timeout=30)
            summary = run_tasks(
                platform,
                manifest,
                benchmark,
                tasks,
                mode,
                output / mode / "tasks",
                servers[mode].port,
                environment_root / mode if environment_root else None,
                budget_s,
                stop_event=stop,
            )
            if summary["controller_error"]:
                stop.set()
            return summary
        except BaseException:
            stop.set()
            barrier.abort()
            raise

    with ThreadPoolExecutor(max_workers=len(modes)) as pool:
        futures = {pool.submit(run_mode, mode): mode for mode in modes}
        try:
            for future in as_completed(futures):
                summaries[futures[future]] = future.result()
        except BaseException:
            stop.set()
            barrier.abort()
            raise
    return [summaries[mode] for mode in modes]


def run_parallel(args, manifest, tasks):
    gpu_by_mode = manifest["pilot"]["gpu_by_mode"]
    if not 1 <= len(args.modes) <= 2:
        raise ValueError("Run one comparison pair at a time, then score and review it")
    gpus = [gpu_by_mode[mode] for mode in args.modes]
    if len(set(gpus)) != len(gpus):
        raise ValueError("Parallel comparison modes require distinct GPUs")
    summaries, failure = [], None
    measured_wall_s = 0.0
    remaining = args.budget_seconds
    budget = None
    try:
        with ExitStack() as stack:
            for gpu in sorted(gpus):
                lock = stack.enter_context(
                    (args.platform / f"gpu-{gpu}.lock").open("a")
                )
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            budget = stack.enter_context(
                Budget(
                    args.budget_ledger,
                    digest(args.manifest),
                    manifest["pilot"]["measurement_budget_s"],
                )
            )
            remaining = min(args.budget_seconds, budget.remaining_s)
            if remaining <= 0:
                raise ValueError("Campaign measurement budget exhausted")
            if args.environment_root is not None:
                for mode in args.modes:
                    preflight_environments(tasks, args.environment_root / mode)
            servers = stack.enter_context(
                paired_services(args.platform, args.output, args.modes, gpu_by_mode)
            )
            with budget.measurement(f"{args.output}:parallel:{','.join(args.modes)}"):
                started = time.monotonic()
                try:
                    summaries = parallel_task_runs(
                        args.platform,
                        manifest,
                        args.benchmark,
                        tasks,
                        args.modes,
                        args.output,
                        servers,
                        args.environment_root,
                        remaining,
                    )
                finally:
                    measured_wall_s = time.monotonic() - started
            remaining = min(args.budget_seconds - measured_wall_s, budget.remaining_s)
            failure = next(
                (
                    row["controller_error"]
                    for row in summaries
                    if row["controller_error"]
                ),
                None,
            )
    except BaseException as exc:
        failure = {"error_type": type(exc).__name__, "error": str(exc)}
    finally:
        if budget is not None and hasattr(budget, "used_s"):
            remaining = min(args.budget_seconds - measured_wall_s, budget.remaining_s)
        for mode in args.modes:
            for task in tasks:
                task_id = task.get("instance_id", task.get("id"))
                terminal = args.output / mode / "tasks" / task_id / "terminal.json"
                if not terminal.exists():
                    write_json(
                        terminal,
                        {
                            "task_id": task_id,
                            "mode": mode,
                            "status": "parallel_controller_error",
                            "error": failure,
                        },
                    )
        write_json(
            args.output / "execution.json",
            {
                "service_execution": "parallel_pair",
                "gpu_by_mode": gpu_by_mode,
                "host_resource_scope": (
                    "shared CPU quota and cgroup memory across both GPUs"
                ),
                "runs": summaries,
                "measurement_wall_s": measured_wall_s,
                "budget_accounting": "union of overlapping mode measurement intervals",
                "remaining_budget_s": max(0.0, remaining),
                "controller_error": failure,
                "planned_modes": args.modes,
                "completed_mode_runs": len(summaries),
            },
        )
    if failure:
        raise SystemExit(
            "Parallel comparison incomplete; inspect execution.json and terminals"
        )
