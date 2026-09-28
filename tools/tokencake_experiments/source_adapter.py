# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read frozen workloads or analyze results using repository-local snapshots."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "analyze"])
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parameters = json.loads(args.input.read_text())
    checkout = args.checkout.resolve()
    sys.path.insert(0, str(checkout))
    if args.command == "freeze":
        from dataset import load_dataset

        dataset = load_dataset(checkout / "workload-dataset.json")
        result = dataset.freeze(parameters["qps"])
    else:
        from analysis import summarize_attempt
        from dataset import load_dataset

        dataset = load_dataset(checkout / "workload-dataset.json")
        result = summarize_attempt(
            Path(parameters["case_dir"]),
            parameters["attempt"],
            parameters["total_e2e_s"],
            parameters["contamination"],
            parameters["state_isolation"],
            dataset=dataset,
        )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
