# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture the immutable inputs to the TokenCake migration experiments."""

import argparse
import csv
import hashlib
import io
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SOURCE_COMMIT = "7a608a4e53ea990b2540c93b4d28cb795b905109"
REFERENCE_COMMIT = "8b6de0eb9a09ef53f20cf06bd4d17ee264b9c2a7"
MOONCAKE_COMMIT = "696c9a14f30ffeacda1707e8f712b7a214460be6"
DATASET_SHA256 = "730f251343121339d87313895add089f2520971319bdc92c282d5497a6425ca0"
MOONCAKE_SHA256 = "37354b69e87f84c3c0162839519041b64b39714adf52bf44eeda37a929a12962"


def command(args: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def repository(path: Path, expected_commit: str | None = None) -> dict[str, Any]:
    commit = command(["git", "rev-parse", "HEAD"], path)
    if expected_commit is not None and commit != expected_commit:
        raise ValueError(f"Unexpected commit at {path}: {commit}")
    return {
        "path": str(path.resolve()),
        "commit": commit,
        "branch": command(["git", "branch", "--show-current"], path),
        "status": command(["git", "status", "--porcelain=v1"], path),
    }


def checked_identity(path: Path, expected_sha256: str) -> dict[str, Any]:
    identity = file_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"Unexpected SHA-256 at {path}: {identity['sha256']}")
    return identity


def capture(args: argparse.Namespace) -> dict[str, Any]:
    root = args.target.resolve()
    source = root.parent / "vllm_agent"
    reference = root.parent / "vllm-note"
    mooncake = root.parent / "Mooncake-TokenCake"
    repositories = {
        "target": repository(root),
        "source": repository(source, SOURCE_COMMIT),
        "native_reference": repository(reference, REFERENCE_COMMIT),
        "mooncake_configuration": repository(mooncake, MOONCAKE_COMMIT),
    }
    repositories["native_reference"]["role"] = "reference-only"
    for name in ("source", "native_reference", "mooncake_configuration"):
        if repositories[name]["status"]:
            raise ValueError(f"Read-only repository has changes: {name}")

    source_files = command(
        [
            "git",
            "ls-files",
            "tools/tokencake_experiments",
            "tools/cpu_offload_vllm_serving_benchmark.py",
            "vllm_serving.py",
            "agent",
            "docs/pkua100_cpu_offload_experiment_handoff.md",
        ],
        source,
    ).splitlines()
    helpers = [file_identity(source / name) for name in source_files]
    # Compare the transferred files to the frozen Git objects, not just each other.
    for name, identity in zip(source_files, helpers):
        committed = subprocess.check_output(
            ["git", "show", f"{SOURCE_COMMIT}:{name}"], cwd=source
        )
        if hashlib.sha256(committed).hexdigest() != identity["sha256"]:
            raise ValueError(f"Transferred helper differs from source commit: {name}")

    gpu_command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,pci.bus_id,memory.total",
        "--format=csv,nounits",
    ]
    gpu_rows = list(csv.DictReader(io.StringIO(command(gpu_command))))
    gpu_rows = [
        {key.strip(): value.strip() for key, value in row.items()} for row in gpu_rows
    ]
    expected_gpus = (
        "GPU-dfe6761a-23f4-c34c-9c03-01da02020e5f",
        "GPU-ce81ab13-d8a0-a49d-c908-f876b8eb2087",
    )
    if tuple(row["uuid"] for row in gpu_rows) != expected_gpus:
        raise ValueError(f"GPU assignment differs from the accepted matrix: {gpu_rows}")
    numa = {
        str(node): Path(f"/sys/devices/system/node/node{node}/cpulist")
        .read_text()
        .strip()
        for node in (0, 1)
    }
    if numa != {"0": "0-35,72-107", "1": "36-71,108-143"}:
        raise ValueError(f"NUMA assignment differs from the accepted matrix: {numa}")

    model = args.model.resolve()
    index = json.loads((model / "model.safetensors.index.json").read_text())
    model_files = sorted(
        set(index["weight_map"].values())
        | {
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "merges.txt",
            "vocab.json",
        }
    )
    original_packages = json.loads(
        command(
            [
                "uv",
                "pip",
                "list",
                "--python",
                str(args.original_environment / "bin/python"),
                "--format=json",
            ]
        )
    )
    return {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "repositories": repositories,
        "official_baseline_commit": command(
            ["git", "rev-parse", "0b3ba88f1^{commit}"], root
        ),
        "paper": {
            "url": "https://arxiv.org/abs/2510.18586v4",
            "revision_date": "2026-08-21",
            "role": "algorithmic context; exact source HEAD resolves differences",
        },
        "transferred_source_helpers": helpers,
        "dataset": checked_identity(
            source / "dataset/agentcodeclean_new.json", DATASET_SHA256
        ),
        "model": {
            "path": str(model),
            "files": [file_identity(model / name) for name in model_files],
        },
        "mooncake_wheel": checked_identity(args.mooncake_wheel, MOONCAKE_SHA256),
        "vllm_wheel": file_identity(args.vllm_wheel),
        "original_environment": {
            "path": str(args.original_environment.resolve()),
            "packages": original_packages,
            "mutation_policy": "read-only; repo-local uv environments own overrides",
        },
        "hardware": {"gpus": gpu_rows, "numa_cpu_sets": numa},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--original-environment", type=Path, required=True)
    parser.add_argument("--mooncake-wheel", type=Path, required=True)
    parser.add_argument("--vllm-wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = capture(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Verified provenance written to {args.output}")


if __name__ == "__main__":
    main()
