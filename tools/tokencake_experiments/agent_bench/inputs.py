# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture official task inputs and a reviewable deterministic pilot manifest."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import yaml

from tools.tokencake_experiments.campaign import (
    ROOT,
    Settings,
    add_settings_arguments,
    settings_from_args,
)
from tools.tokencake_experiments.provenance import repository, resolve_model

from .transport import write_json


def resource_limit(name):
    path = Path("/sys/fs/cgroup") / name
    try:
        return path.read_text().strip()
    except OSError:
        return None


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ordered_ids(ids: list[str], seed: str) -> list[str]:
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate task IDs")
    return sorted(
        ids, key=lambda item: hashlib.sha256(f"{seed}:{item}".encode()).hexdigest()
    )


def swe_instances(task_repo: Path) -> list[dict]:
    """Read the official task-repo format without requiring Docker dependencies."""
    instances = []
    for path in sorted((task_repo / "tasks").glob("*/task.yaml")):
        entry = yaml.safe_load(path.read_text())
        if (
            "SWE-bench/SWE-bench_Verified" not in entry.get("datasets", [])
            or entry["split"] != "test"
        ):
            continue
        for field, name in (
            ("problem_statement", "problem_statement.md"),
            ("patch", "gold.patch"),
            ("test_patch", "test.patch"),
            ("eval_script", "eval.sh"),
        ):
            with (path.parent / name).open(newline="") as stream:
                entry[field] = stream.read()
        entry.update(json.loads((path.parent / "tests.json").read_text()))
        instances.append(entry)
    return instances


def capture(
    platform: Path,
    output: Path,
    seed: str,
    workers: int,
    status: str,
    dependencies: Path | None = None,
    execution: str = "sequential",
    settings: Settings | None = None,
):
    if execution not in ("sequential", "parallel_pair"):
        raise ValueError(f"Unknown service execution: {execution}")
    settings = settings or Settings()
    if execution == "parallel_pair" and len(settings.gpus) < 2:
        raise ValueError("Parallel pairs require at least two selected GPUs")
    output.mkdir(parents=True, exist_ok=False)
    model = resolve_model(settings.model)
    sources = platform / "sources"
    repos = {
        name: repository(sources / name)
        for name in ("gorilla", "mini-swe-agent", "SWE-bench", "swe-bench-tasks")
    }
    swe = swe_instances(sources / "swe-bench-tasks")
    swe_file = output / "swe-verified.json"
    write_json(swe_file, swe)
    bfcl_root = sources / "gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"
    bfcl_categories = {}
    for category in ("multi_turn_base", "multi_turn_long_context"):
        files = list(bfcl_root.glob(f"BFCL_*_{category}.json"))
        if not files:
            continue
        files.sort()
        entries = read_jsonl(files[0])
        ids = [entry["id"] for entry in entries]
        bfcl_categories[category] = {
            "path": str(files[0]),
            "sha256": digest(files[0]),
            "answers_path": str(bfcl_root / "possible_answer" / files[0].name),
            "ids": ids,
            "probe_ids": ordered_ids(ids, seed)[:2],
        }
    preset = (
        sources / "mini-swe-agent/src/minisweagent/config/benchmarks/swebench_xml.yaml"
    )
    manifest = {
        "schema_version": 1,
        "status": status,
        "selection": {"seed": seed, "rule": 'SHA256(seed + ":" + instance_id)'},
        "repositories": repos,
        "target": repository(ROOT),
        "baseline": repository(ROOT),
        "model_path": model,
        "bfcl": bfcl_categories,
        "swe": {
            "path": str(swe_file),
            "sha256": digest(swe_file),
            "dataset": "SWE-bench/SWE-bench_Verified",
            "split": "test",
            "source": "official GitHub task repository",
            "ids": [entry["instance_id"] for entry in swe],
            "pilot_ids": ordered_ids([entry["instance_id"] for entry in swe], seed)[
                :20
            ],
            "preset_path": str(preset),
        },
        "pilot": {
            "modes_in_order": ["agent_offload", "base", "agent", "offload"],
            "workers": workers,
            "gpu": settings.gpus[0],
            "service_execution": execution,
            "gpu_by_mode": {
                mode: settings.gpus[index % len(settings.gpus)]
                if execution == "parallel_pair"
                else settings.gpus[0]
                for index, mode in enumerate(
                    ("base", "agent_offload", "agent", "offload")
                )
            },
            "arrival": "closed_loop_in_manifest_order",
            "measurement_budget_s": 12 * 3600,
        },
        "service": {
            "dtype": settings.dtype,
            "max_model_len": settings.max_model_len,
            "gpu_memory_utilization": settings.gpu_memory_utilization,
            "max_num_batched_tokens": 8192,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "cpu_kv_gib": settings.cpu_offload_gib,
            "python": settings.python,
            "cpus": settings.cpus,
        },
        "resources": {
            "cgroup_memory_max": resource_limit("memory.max"),
            "cgroup_cpu_max": resource_limit("cpu.max"),
            "library_threads_per_client_and_tool": 1,
            "service_execution": (
                "independent services on selected GPUs; shared CPU quota and memory"
                if execution == "parallel_pair"
                else "one selected GPU service at a time"
            ),
        },
        "task_environments": {
            "primary_manager": "uv",
            "compatibility_manager": (
                "server Miniconda when uv cannot reproduce the recipe"
            ),
            "conda_executable": os.environ.get("TC_BENCH_CONDA")
            or shutil.which("conda"),
            "storage": str(platform / "environments"),
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--status", default="ready", help="Optional run label")
    parser.add_argument("--dependencies", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--execution", choices=("sequential", "parallel_pair"), default="sequential"
    )
    add_settings_arguments(parser)
    args = parser.parse_args()
    manifest = capture(
        args.platform.resolve(),
        args.output.resolve(),
        args.seed,
        args.workers,
        args.status,
        args.dependencies,
        args.execution,
        settings=settings_from_args(args),
    )
    print(
        json.dumps(
            {
                "path": str(args.output),
                "swe_pilot": manifest["swe"]["pilot_ids"],
                "bfcl_probe": {
                    key: value["probe_ids"] for key, value in manifest["bfcl"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
