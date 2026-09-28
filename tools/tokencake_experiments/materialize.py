# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Snapshot the current repository's JSON workload, client, and analyzer."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
WORKLOAD_PROFILES = ("frozen", "continuation", "conversation", "conversation-tools")
DATASET_NAME = "workload-dataset.json"
HELPERS = ("dataset.py", "dataset_client.py", "analysis.py")


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def git(root, *args, input=None):
    return subprocess.run(
        ["git", "-C", str(root), *args],
        input=input,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def materialize(
    source, destination, *, workload_profile="frozen", workload_dataset=None
):
    from tools.tokencake_experiments.dataset import load_dataset

    if workload_profile not in WORKLOAD_PROFILES:
        raise ValueError(f"Unknown workload profile: {workload_profile}")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    package = source / "tools/tokencake_experiments"
    dataset_path = Path(
        workload_dataset or package / "datasets" / f"{workload_profile}.json"
    ).resolve()
    load_dataset(dataset_path)
    original = {name: digest(package / name) for name in HELPERS}
    from tools.tokencake_experiments.provenance import repository

    revision = repository(source)["commit"]
    destination.mkdir(parents=True, exist_ok=False)
    for name in HELPERS:
        shutil.copyfile(package / name, destination / name)
    shutil.copyfile(dataset_path, destination / DATASET_NAME)
    return {
        "source_revision": revision,
        "source": str(source),
        "checkout": str(destination),
        "workload_profile": workload_profile,
        "workload_dataset": str(dataset_path),
        "workload_dataset_sha256": digest(destination / DATASET_NAME),
        "source_helpers": original,
        "materialized_helpers": {
            name: digest(destination / name) for name in (*HELPERS, DATASET_NAME)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
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


if __name__ == "__main__":
    main()
