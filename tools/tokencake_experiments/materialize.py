# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Snapshot a JSON workload, its client, and frozen analysis helpers."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

SOURCE_REVISION = "7a608a4e53ea990b2540c93b4d28cb795b905109"
PACKAGE = Path(__file__).resolve().parent
WORKLOAD_PROFILES = ("frozen", "continuation", "conversation", "conversation-tools")
DATASET_NAME = "workload-dataset.json"
HELPERS = (
    "vllm_serving.py",
    "agent",
    "tools/tokencake_experiments/__init__.py",
    "tools/tokencake_experiments/analysis.py",
    "tools/tokencake_experiments/catalog.py",
    "tools/tokencake_experiments/commands.py",
    "tools/tokencake_experiments/evidence.py",
    "tools/tokencake_experiments/schema.py",
    "tools/tokencake_experiments/launch_server.py",
)


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def git(root: Path, *args: str, input: str | None = None) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        input=input,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def materialize(
    source: Path,
    destination: Path,
    *,
    workload_profile: str = "frozen",
    workload_dataset: Path | None = None,
) -> dict:
    from tools.tokencake_experiments.dataset import load_dataset

    if workload_profile not in WORKLOAD_PROFILES:
        raise ValueError(f"Unknown workload profile: {workload_profile}")
    dataset_path = (
        workload_dataset or PACKAGE / "datasets" / f"{workload_profile}.json"
    ).resolve()
    dataset = load_dataset(dataset_path)
    if dataset.profile != workload_profile:
        raise ValueError("Dataset profile does not match workload_profile")
    source, destination = source.resolve(), destination.resolve()
    revision = git(source, "rev-parse", "HEAD")
    if revision != SOURCE_REVISION or git(source, "status", "--porcelain"):
        raise ValueError("Source must be clean at the frozen revision")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    git(
        source,
        "clone",
        "--shared",
        "--no-checkout",
        "--quiet",
        str(source),
        str(destination),
    )
    patterns = "\n".join(
        "/" + name + ("/" if name == "agent" else "") for name in HELPERS
    )
    git(
        destination,
        "sparse-checkout",
        "set",
        "--no-cone",
        "--stdin",
        input=patterns + "\n",
    )
    git(destination, "checkout", "--detach", SOURCE_REVISION)
    original = {
        name: digest(destination / name)
        for name in git(destination, "ls-files").splitlines()
        if (destination / name).is_file()
    }
    for name in ("dataset.py", "dataset_client.py"):
        shutil.copyfile(PACKAGE / name, destination / name)
    shutil.copyfile(dataset_path, destination / DATASET_NAME)
    if git(source, "status", "--porcelain"):
        raise RuntimeError("Source worktree changed during materialization")
    return {
        "source_revision": revision,
        "source": str(source),
        "checkout": str(destination),
        "workload_profile": workload_profile,
        "workload_dataset": str(dataset_path),
        "workload_dataset_sha256": digest(destination / DATASET_NAME),
        "source_helpers": original,
        "materialized_helpers": {
            name: digest(destination / name)
            for name in (*original, "dataset.py", "dataset_client.py", DATASET_NAME)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--workload-dataset", type=Path)
    parser.add_argument(
        "--workload-profile", choices=WORKLOAD_PROFILES, default="frozen"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(
        args.source,
        args.destination,
        workload_profile=args.workload_profile,
        workload_dataset=args.workload_dataset,
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Materialized launcher: {args.destination}")


if __name__ == "__main__":
    main()
