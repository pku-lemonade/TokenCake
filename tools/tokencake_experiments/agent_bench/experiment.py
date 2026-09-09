# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a frozen real-agent task set against fresh model services."""

import argparse
import fcntl
import json
import signal
import subprocess
import time
from collections import Counter
from contextlib import suppress
from pathlib import Path

import psutil

from .budget import Budget
from .inputs import ROOT, digest, read_jsonl, repository, verify
from .service import Service, worker_environment
from .swe_local import restore_prepared_state, task_environment
from .transport import MODES, write_json

CLEANUP_GRACE_S = 110.0


def terminate(process: subprocess.Popen):
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    # Stop the owner after its tool children so they cannot survive a deadline.
    for child in reversed(children):
        with suppress(psutil.NoSuchProcess):
            child.kill()
    try:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def tasks_for(manifest: dict, benchmark: str, subset: str) -> list[dict]:
    if benchmark == "bfcl":
        tasks = []
        for category in manifest["bfcl"].values():
            path = Path(category["path"])
            if digest(path) != category["sha256"]:
                raise ValueError("BFCL dataset changed after freezing")
            by_id = {entry["id"]: entry for entry in read_jsonl(path)}
            ids = category["ids"] if subset == "full" else category["probe_ids"]
            tasks.extend(by_id[task_id] for task_id in ids)
        return tasks
    path = Path(manifest["swe"]["path"])
    if digest(path) != manifest["swe"]["sha256"]:
        raise ValueError("SWE dataset changed after freezing")
    by_id = {entry["instance_id"]: entry for entry in json.loads(path.read_text())}
    ids = manifest["swe"]["ids"] if subset == "full" else manifest["swe"]["pilot_ids"]
    return [by_id[task_id] for task_id in ids]


def run_tasks(
    platform: Path,
    manifest: dict,
    benchmark: str,
    tasks: list[dict],
    mode: str,
    output: Path,
    port: int,
    environment_root: Path | None,
    budget_s: float,
    stop_event=None,
):
    output.mkdir(parents=True, exist_ok=False)
    workers = manifest["pilot"]["workers"]
    if workers < 1 or budget_s <= 0:
        raise ValueError("Worker count and measurement budget must be positive")
    started = time.time()
    budget_end = started + budget_s
    active = {}
    results = []
    next_index = 0
    failure = None
    launching = None
    try:
        while active or next_index < len(tasks):
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError("Parallel comparison cancelled")
            while (
                len(active) < workers
                and next_index < len(tasks)
                and time.time() < budget_end
                and (stop_event is None or not stop_event.is_set())
            ):
                task = tasks[next_index]
                task_id = task.get("instance_id", task.get("id"))
                next_index += 1
                task_output = output / task_id
                task_output.mkdir()
                # Pilot uses closed-loop arrivals: release the next application
                # when a worker slot opens. Save the actual release trace.
                arrived = time.time()
                allowance = 1800 if benchmark == "bfcl" else 3600
                context = {
                    "task_id": task_id,
                    "agent_type": "bfcl" if benchmark == "bfcl" else "swe_coder",
                    "mode": mode,
                    "arrived_at": arrived,
                    "arrival_offset_s": arrived - started,
                    "deadline_at": min(arrived + allowance, budget_end),
                }
                launching = (task_id, task_output, context)
                spec = {
                    "benchmark": benchmark,
                    "context": context,
                    "base_url": f"http://127.0.0.1:{port}/v1",
                    # Only the problem goes to the SWE worker, never the gold
                    # patch, test patch, F2P/P2P, hints, or grading script.
                    "task": task
                    if benchmark == "bfcl"
                    else {"problem_statement": task["problem_statement"]},
                }
                if benchmark == "swe":
                    directory = environment_root / task_id
                    preparation = json.loads(
                        (directory / "preparation.json").read_text()
                    )
                    if preparation["status"] != "prepared":
                        raise ValueError(f"Task environment not prepared: {task_id}")
                    spec.update(
                        {
                            "preset": manifest["swe"]["preset_path"],
                            "task_directory": str(directory),
                            "task_environment": task_environment(platform, directory),
                        }
                    )
                write_json(task_output / "spec.json", spec)
                log = (task_output / "worker.log").open("x")
                command = [
                    str(platform / ".venv/bin/python"),
                    "-m",
                    "tools.tokencake_experiments.agent_bench.worker",
                    "--spec",
                    str(task_output / "spec.json"),
                    "--output",
                    str(task_output),
                ]
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=worker_environment(platform, task_output),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except BaseException:
                    log.close()
                    raise
                active[task_id] = (process, log, task_output, context)
                launching = None
            for task_id, (process, log, task_output, context) in list(active.items()):
                code = process.poll()
                expired = time.time() > context["deadline_at"] + CLEANUP_GRACE_S
                # Give outstanding stall_finished acknowledgements a bounded
                # cleanup interval, while retaining the original task deadline.
                if code is None and not expired:
                    continue
                if code is None:
                    terminate(process)
                    code = process.returncode
                log.close()
                path = task_output / "result.json"
                try:
                    result = json.loads(path.read_text())
                    if result.get("task_id") != task_id or not result.get("status"):
                        raise ValueError("Invalid worker terminal record")
                    if code:
                        result = result | {
                            "application_status": result["status"],
                            "status": "worker_cleanup_error",
                            "exit_code": code,
                        }
                except (OSError, ValueError):
                    result = {
                        "task_id": task_id,
                        "mode": mode,
                        "status": "supervisor_timeout" if expired else "worker_error",
                        "exit_code": code,
                        "arrived_at": context["arrived_at"],
                        "ended_at": time.time(),
                        "e2e_s": time.time() - context["arrived_at"],
                    }
                write_json(task_output / "terminal.json", result)
                results.append(result)
                del active[task_id]
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "benchmark": benchmark,
                            "completed": len(results),
                            "total": len(tasks),
                            "task_id": task_id,
                            "status": result["status"],
                        }
                    ),
                    flush=True,
                )
            if time.time() >= budget_end and not active:
                break
            time.sleep(0.2)
    except BaseException as exc:
        failure = {"error_type": type(exc).__name__, "error": str(exc)}
        if launching is not None:
            task_id, task_output, context = launching
            result = {
                "task_id": task_id,
                "mode": mode,
                "status": "worker_launch_error",
                "arrived_at": context["arrived_at"],
                "ended_at": time.time(),
                "e2e_s": time.time() - context["arrived_at"],
                "error": failure,
            }
            write_json(task_output / "terminal.json", result)
            results.append(result)
    finally:
        for task_id, (process, log, task_output, context) in active.items():
            cleanup_error = None
            try:
                terminate(process)
            except Exception as exc:
                cleanup_error = str(exc)
            finally:
                log.close()
            result = {
                "task_id": task_id,
                "mode": mode,
                "status": "controller_cancelled",
                "arrived_at": context["arrived_at"],
                "ended_at": time.time(),
                "e2e_s": time.time() - context["arrived_at"],
                "controller_error": failure,
                "cleanup_error": cleanup_error,
            }
            write_json(task_output / "terminal.json", result)
            results.append(result)
    recorded_ids = {result["task_id"] for result in results}
    for task in tasks:
        task_id = task.get("instance_id", task.get("id"))
        if task_id in recorded_ids:
            continue
        result = {
            "task_id": task_id,
            "mode": mode,
            "status": "not_started_controller_error"
            if failure
            else "not_started_budget_exhausted",
            "controller_error": failure,
        }
        write_json(output / task_id / "terminal.json", result)
        results.append(result)
    summary = {
        "mode": mode,
        "benchmark": benchmark,
        "started_at": started,
        "ended_at": time.time(),
        "wall_s": time.time() - started,
        "planned": len(tasks),
        "recorded": len(results),
        "statuses": dict(Counter(r["status"] for r in results)),
        "arrival_policy": manifest["pilot"]["arrival"],
        "workers": workers,
        "controller_error": failure,
    }
    write_json(output / "execution.json", summary)
    return summary


def record_unstarted(tasks: list[dict], mode: str, output: Path, status: str):
    for task in tasks:
        task_id = task.get("instance_id", task.get("id"))
        write_json(
            output / task_id / "terminal.json",
            {
                "task_id": task_id,
                "mode": mode,
                "status": status,
            },
        )


def preflight_environments(tasks: list[dict], environment_root: Path):
    for task in tasks:
        directory = environment_root / task["instance_id"]
        preparation = json.loads((directory / "preparation.json").read_text())
        if (
            preparation["status"] != "prepared"
            or preparation["base_commit"] != task["base_commit"]
        ):
            raise ValueError(
                f"Task environment does not match the dataset: {directory}"
            )
        restore_prepared_state(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("bfcl", "swe"), required=True)
    parser.add_argument("--subset", choices=("pilot", "full"), required=True)
    parser.add_argument("--modes", choices=MODES, nargs="+", required=True)
    parser.add_argument("--environment-root", type=Path)
    parser.add_argument("--budget-seconds", type=float, required=True)
    parser.add_argument("--budget-ledger", type=Path, required=True)
    args = parser.parse_args()
    if args.budget_seconds <= 0 or len(set(args.modes)) != len(args.modes):
        parser.error("Use a positive budget and distinct comparison modes")
    manifest = json.loads(args.manifest.read_text())
    if manifest["status"] != "confirmed":
        raise ValueError("Review and confirm the candidate manifest before measurement")
    verify(manifest)
    if args.benchmark == "swe" and args.environment_root is None:
        parser.error("SWE needs a separate prepared environment root for every mode")
    if args.output.exists():
        raise FileExistsError("Every experiment needs a fresh output directory")
    args.output.mkdir(parents=True)
    tasks = tasks_for(manifest, args.benchmark, args.subset)
    config = vars(args) | {
        "manifest_sha256": digest(args.manifest),
        "target": repository(ROOT),
        "adapter_sha256": {
            str(path.relative_to(ROOT)): digest(path)
            for path in Path(__file__).parent.glob("*.py")
        },
    }
    write_json(
        args.output / "experiment.json", json.loads(json.dumps(config, default=str))
    )
    if manifest["pilot"].get("service_execution") == "parallel_pair":
        from .parallel import run_parallel

        run_parallel(args, manifest, tasks)
        return
    remaining = args.budget_seconds
    summaries = []
    failure = None
    with (
        (args.platform / "gpu-0.lock").open("a") as lock,
        Budget(
            args.budget_ledger,
            digest(args.manifest),
            manifest["pilot"]["measurement_budget_s"],
        ) as budget,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for mode in args.modes:
            remaining = min(remaining, budget.remaining_s)
            if remaining <= 0 or failure:
                record_unstarted(
                    tasks,
                    mode,
                    args.output / mode / "tasks",
                    "not_started_controller_error"
                    if failure
                    else "not_started_budget_exhausted",
                )
                continue
            environment_root = (
                args.environment_root / mode if args.environment_root else None
            )
            try:
                if environment_root is not None:
                    preflight_environments(tasks, environment_root)
                with Service(
                    args.platform, args.output / mode / "service", mode
                ) as service:
                    with budget.measurement(f"{args.output}:{mode}"):
                        summary = run_tasks(
                            args.platform,
                            manifest,
                            args.benchmark,
                            tasks,
                            mode,
                            args.output / mode / "tasks",
                            service.port,
                            environment_root,
                            remaining,
                        )
                    remaining -= summary["wall_s"]
                    summaries.append(summary)
                    failure = summary["controller_error"]
            except BaseException as exc:
                failure = {"error_type": type(exc).__name__, "error": str(exc)}
                for task in tasks:
                    task_id = task.get("instance_id", task.get("id"))
                    path = args.output / mode / "tasks" / task_id / "terminal.json"
                    if not path.exists():
                        write_json(
                            path,
                            {
                                "task_id": task_id,
                                "mode": mode,
                                "status": "environment_or_service_error",
                                "error": failure,
                            },
                        )
    write_json(
        args.output / "execution.json",
        {
            "runs": summaries,
            "remaining_budget_s": remaining,
            "controller_error": failure,
            "planned_modes": args.modes,
            "completed_mode_runs": len(summaries),
        },
    )
    if failure:
        raise SystemExit("Experiment incomplete; see execution.json and task terminals")


if __name__ == "__main__":
    main()
