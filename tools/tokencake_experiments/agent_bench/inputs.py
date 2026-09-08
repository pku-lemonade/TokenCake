# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture official task inputs and a reviewable deterministic pilot manifest."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from .transport import write_json

ROOT = Path(__file__).resolve().parents[3]
MODEL = Path("/root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def repository(path: Path) -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=path, text=True).strip()

    return {
        "path": str(path),
        "commit": git("rev-parse", "HEAD"),
        "tracked_status": git("status", "--porcelain=v1", "--untracked-files=no"),
    }


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
    if len(instances) != 500:
        raise ValueError(f"Expected Verified/test 500, found {len(instances)}")
    return instances


def capture(
    platform: Path,
    output: Path,
    seed: str,
    workers: int,
    status: str,
    dependencies: Path,
    execution: str = "sequential",
):
    if execution not in ("sequential", "parallel_pair"):
        raise ValueError(f"Unknown service execution: {execution}")
    sources = platform / "sources"
    repos = {
        name: repository(sources / name)
        for name in ("gorilla", "mini-swe-agent", "SWE-bench", "swe-bench-tasks")
    }
    if any(repo["tracked_status"] for repo in repos.values()):
        raise ValueError("Official source checkout has tracked modifications")
    swe = swe_instances(sources / "swe-bench-tasks")
    swe_file = output / "swe-verified.json"
    write_json(swe_file, swe)
    bfcl_root = sources / "gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"
    bfcl_categories = {}
    for category in ("multi_turn_base", "multi_turn_long_context"):
        files = list(bfcl_root.glob(f"BFCL_*_{category}.json"))
        if len(files) != 1:
            raise ValueError(f"Ambiguous BFCL dataset version: {files}")
        entries = read_jsonl(files[0])
        ids = [entry["id"] for entry in entries]
        if len(ids) != 200:
            raise ValueError(f"Unexpected BFCL {category} count: {len(ids)}")
        bfcl_categories[category] = {
            "path": str(files[0]),
            "sha256": digest(files[0]),
            "answers_path": str(bfcl_root / "possible_answer" / files[0].name),
            "answers_sha256": digest(bfcl_root / "possible_answer" / files[0].name),
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
        "baseline": repository(ROOT / ".venv/baseline"),
        "model_path": str(MODEL),
        "model_identity_path": str(platform / "manifests/model-identity.json"),
        "model_identity_sha256": digest(platform / "manifests/model-identity.json"),
        "dependency_identity_path": str(dependencies),
        "dependency_identity_sha256": digest(dependencies),
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
            "preset_sha256": digest(preset),
        },
        "pilot": {
            "modes_in_order": ["agent_offload", "base", "agent", "offload"],
            "workers": workers,
            "gpu": 0,
            "service_execution": execution,
            "gpu_by_mode": {"base": 0, "agent_offload": 1, "agent": 0, "offload": 1}
            if execution == "parallel_pair"
            else {mode: 0 for mode in ("base", "agent_offload", "agent", "offload")},
            "arrival": "closed_loop_in_manifest_order",
            "measurement_budget_s": 12 * 3600,
        },
        "service": {
            "dtype": "bfloat16",
            "max_model_len": 32768,
            "gpu_memory_utilization": 0.5,
            "max_num_batched_tokens": 8192,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "cpu_kv_gib": 100,
        },
        "resources": {
            "cgroup_memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
            "cgroup_cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
            "library_threads_per_client_and_tool": 1,
            "service_execution": (
                "independent services on GPU 0 and GPU 1; shared CPU quota and memory"
                if execution == "parallel_pair"
                else "one GPU 0 service at a time"
            ),
        },
        "task_environments": {
            "primary_manager": "uv",
            "compatibility_manager": (
                "server Miniconda when uv cannot reproduce the recipe"
            ),
            "conda_executable": "/root/miniconda3/bin/conda",
            "storage": str(platform / "environments"),
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def verify(manifest: dict):
    for expected in [
        *manifest["repositories"].values(),
        manifest["baseline"],
        manifest["target"],
    ]:
        actual = repository(Path(expected["path"]))
        if (
            actual["commit"] != expected["commit"]
            or actual["tracked_status"] != expected["tracked_status"]
        ):
            raise ValueError(f"Frozen code identity changed: {expected['path']}")
    preset = Path(manifest["swe"]["preset_path"])
    if digest(preset) != manifest["swe"]["preset_sha256"]:
        raise ValueError("Official mini-swe-agent preset changed")
    for key in ("model", "dependency"):
        path = Path(manifest[f"{key}_identity_path"])
        if digest(path) != manifest[f"{key}_identity_sha256"]:
            raise ValueError(f"Frozen {key} identity changed")
    model = json.loads(Path(manifest["model_identity_path"]).read_text())
    if Path(model["model"]) != MODEL:
        raise ValueError("Model path differs from the agreed server model")
    for identity in model["files"]:
        path = MODEL / identity["name"]
        if (
            path.stat().st_size != identity["size"]
            or digest(path) != identity["sha256"]
        ):
            raise ValueError(f"Frozen model file changed: {path}")
    dependencies = json.loads(Path(manifest["dependency_identity_path"]).read_text())
    for identity in dependencies.values():
        freeze = Path(identity["freeze"])
        if digest(freeze) != identity["freeze_sha256"]:
            raise ValueError(f"Dependency lock changed: {freeze}")
        current = subprocess.check_output(
            ["uv", "pip", "freeze", "--python", identity["python"]], text=True
        )
        if current != freeze.read_text():
            raise ValueError(
                f"Installed dependencies differ from the lock: {identity['python']}"
            )
    for filename, key in (
        ("memory.max", "cgroup_memory_max"),
        ("cpu.max", "cgroup_cpu_max"),
    ):
        if (
            Path("/sys/fs/cgroup", filename).read_text().strip()
            != manifest["resources"][key]
        ):
            raise ValueError(f"Resource limit differs from the manifest: {filename}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--status", choices=("candidate", "confirmed"), required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument(
        "--execution", choices=("sequential", "parallel_pair"), default="sequential"
    )
    args = parser.parse_args()
    manifest = capture(
        args.platform.resolve(),
        args.output.resolve(),
        args.seed,
        args.workers,
        args.status,
        args.dependencies.resolve(),
        args.execution,
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
