# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record the actual checkout, interpreter, model, and data used by a run."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tools.tokencake_experiments.materialize import ROOT, git


def repository(path):
    path = Path(path).resolve()
    try:
        commit = git(path, "rev-parse", "HEAD")
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"path": str(path), "commit": commit}


def resolve_model(model):
    path = Path(model).expanduser()
    return str(path.resolve()) if path.is_dir() else str(model)


def capture(args):
    return {
        "schema_version": 3,
        "repository": repository(args.target),
        "model": resolve_model(args.model),
        "dataset": str(args.dataset),
        "environment": {"python": str(args.python)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=ROOT)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        json.dump(capture(args), stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
