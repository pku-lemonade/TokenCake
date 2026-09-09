# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare SWE environments and check both original and reference test behavior."""

import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

from .inputs import ROOT, digest
from .swe_local import grade, prepare, restore_prepared_state
from .transport import write_json


def reference_checks(task: dict, original: dict, reference: dict) -> dict:
    from swebench.harness.grading import _resolve_case, test_passed

    parsed = original.get("parsed_tests", {})
    # Use the frozen official matching rule for truncated parametrized IDs.
    # Missing F2P cases still cannot establish a working baseline environment.
    return {
        "original_test_output_found": original.get("test_output_found", False),
        "original_f2p_failed": all(
            (key := _resolve_case(name, parsed)) is not None
            and parsed[key] in ("FAILED", "ERROR")
            for name in task["FAIL_TO_PASS"]
        ),
        "original_p2p_passed": all(
            test_passed(name, parsed) for name in task["PASS_TO_PASS"]
        ),
        "reference_resolved": reference.get("resolved", False),
        "original_not_resolved": not original.get("resolved", True),
    }


def validate(spec: dict, output: Path) -> dict:
    platform, directory = Path(spec["platform"]), Path(spec["directory"])
    task = spec["task"]
    try:
        preparation = prepare(
            platform,
            platform / "sources/swe-bench-tasks/tasks" / task["instance_id"],
            directory,
            manager=spec.get("environment_manager", "uv"),
        )
        if spec.get("prepare_only", False):
            return {"status": "prepared", "preparation": preparation}
        reports = {}
        for label, patch in (("original", ""), ("reference", task["patch"])):
            reports[label] = grade(
                platform,
                task,
                directory,
                {
                    "instance_id": task["instance_id"],
                    "model_patch": patch,
                    "model_name_or_path": "environment-validation",
                },
                output / label,
            )
        original, reference = reports["original"], reports["reference"]
        checks = reference_checks(task, original, reference)
        restore_prepared_state(directory)
        return {
            "status": "validated"
            if all(checks.values())
            else "reference_validation_failed",
            "checks": checks,
            "preparation": preparation,
            "reports": {key: str(output / key / "report.json") for key in reports},
        }
    except Exception as exc:
        return {
            "status": "environment_error",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }


def run(
    platform: Path,
    manifest_path: Path,
    directory: Path,
    output: Path,
    jobs: int,
    subset: str,
    ids: list[str] | None,
    prepare_only: bool = False,
    manager: str = "uv",
):
    from .experiment import tasks_for, terminate
    from .service import worker_environment

    manifest = json.loads(manifest_path.read_text())
    tasks = tasks_for(manifest, "swe", subset)
    if ids is not None:
        selected = set(ids)
        if len(selected) != len(ids) or not selected <= {
            task["instance_id"] for task in tasks
        }:
            raise ValueError(
                "Requested environment IDs must be distinct "
                "members of the frozen selection"
            )
        tasks = [task for task in tasks if task["instance_id"] in selected]
    if (
        jobs < 1
        or output.exists()
        or any((directory / task["instance_id"]).exists() for task in tasks)
    ):
        raise ValueError(
            "Use fresh selected task/output directories and a positive setup job count"
        )
    directory.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True)
    write_json(
        output / "configuration.json",
        {
            "role": "environment_preparation_not_benchmark"
            if prepare_only
            else "environment_validation_not_benchmark",
            "manifest": str(manifest_path),
            "manifest_sha256": digest(manifest_path),
            "environment_root": str(directory),
            "ids": [task["instance_id"] for task in tasks],
            "setup_jobs": jobs,
            "reference_validation": not prepare_only,
            "environment_manager": manager,
        },
    )
    active, results = {}, []
    next_index = 0
    failure = None
    try:
        while active or next_index < len(tasks):
            while len(active) < jobs and next_index < len(tasks):
                task = tasks[next_index]
                next_index += 1
                task_id = task["instance_id"]
                task_output = output / task_id
                spec = {
                    "platform": str(platform),
                    "directory": str(directory / task_id),
                    "task": task,
                    "prepare_only": prepare_only,
                    "environment_manager": manager,
                }
                write_json(task_output / "spec.json", spec)
                log = (task_output / "worker.log").open("x")
                try:
                    process = subprocess.Popen(
                        [
                            str(platform / "environments/grading/.venv/bin/python"),
                            "-m",
                            "tools.tokencake_experiments.agent_bench.environments",
                            "--one",
                            str(task_output / "spec.json"),
                            "--output",
                            str(task_output),
                        ],
                        cwd=ROOT,
                        env=worker_environment(platform, task_output),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except BaseException:
                    log.close()
                    raise
                active[task_id] = (process, log, task_output)
            for task_id, (process, log, task_output) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                path = task_output / "validation.json"
                result = (
                    json.loads(path.read_text())
                    if path.exists()
                    else {"status": "validation_worker_error", "exit_code": code}
                )
                results.append({"task_id": task_id, **result})
                del active[task_id]
                print(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "status": result["status"],
                            "completed": len(results),
                            "total": len(tasks),
                        }
                    ),
                    flush=True,
                )
            time.sleep(0.2)
    except BaseException as exc:
        failure = {"error": str(exc), "error_type": type(exc).__name__}
    finally:
        for process, log, _ in active.values():
            try:
                terminate(process)
            finally:
                log.close()
    completed = {item["task_id"] for item in results}
    for task in tasks:
        if task["instance_id"] not in completed:
            results.append(
                {
                    "task_id": task["instance_id"],
                    "status": "validation_cancelled",
                    "error": failure,
                }
            )
    write_json(
        output / "summary.json",
        {
            "planned": len(tasks),
            "recorded": len(results),
            "controller_error": failure,
            "statuses": dict(Counter(result["status"] for result in results)),
            "results": results,
        },
    )
    if failure:
        raise SystemExit("Environment validation interrupted; see summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one", type=Path)
    parser.add_argument("--platform", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setup-jobs", type=int)
    parser.add_argument("--subset", choices=("pilot", "full"))
    parser.add_argument("--ids", nargs="+")
    parser.add_argument(
        "--environment-manager", choices=("uv", "miniconda"), default="uv"
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare independent agent workspaces without running reference grading",
    )
    args = parser.parse_args()
    if args.one:
        spec = json.loads(args.one.read_text())
        started = time.time()
        result = validate(spec, args.output)
        write_json(
            args.output / "validation.json",
            {
                "task_id": spec["task"]["instance_id"],
                "started_at": started,
                "ended_at": time.time(),
                "duration_s": time.time() - started,
                **result,
            },
        )
    else:
        if not all(
            (args.platform, args.manifest, args.directory, args.setup_jobs, args.subset)
        ):
            parser.error(
                "Provide platform, manifest, directory, setup-jobs, and subset"
            )
        run(
            args.platform,
            args.manifest,
            args.directory,
            args.output,
            args.setup_jobs,
            args.subset,
            args.ids,
            args.prepare_only,
            args.environment_manager,
        )


if __name__ == "__main__":
    main()
