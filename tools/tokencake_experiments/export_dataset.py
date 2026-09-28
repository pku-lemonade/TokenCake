# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export validated copies of the current repository's fixed JSON workloads."""

import argparse
import json
import shutil
from pathlib import Path

from tools.tokencake_experiments.dataset import load_dataset
from tools.tokencake_experiments.materialize import ROOT, WORKLOAD_PROFILES, digest
from tools.tokencake_experiments.provenance import repository


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    datasets = source / "tools/tokencake_experiments/datasets"
    for profile in WORKLOAD_PROFILES:
        if load_dataset(datasets / f"{profile}.json").profile != profile:
            raise ValueError(f"Dataset profile mismatch: {profile}")
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for profile in WORKLOAD_PROFILES:
        target = args.output / f"{profile}.json"
        shutil.copyfile(datasets / target.name, target)
        hashes[target.name] = digest(target)
    (args.output / "export.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "revision": repository(source)["commit"],
                "files": hashes,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
