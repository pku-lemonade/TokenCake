# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Materialize immutable source helpers and a narrowly patched target client."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

SOURCE_REVISION = "7a608a4e53ea990b2540c93b4d28cb795b905109"
PATCH = Path(__file__).with_name("launcher.patch")
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


def materialize(source: Path, destination: Path, *, patched: bool = True) -> dict:
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
    if patched:
        git(destination, "apply", "--check", "--recount", "--unidiff-zero", str(PATCH))
        git(destination, "apply", "--recount", "--unidiff-zero", str(PATCH))
    modified = git(destination, "diff", "--name-only").splitlines()
    if modified != (["vllm_serving.py"] if patched else []):
        raise RuntimeError(f"Unexpected launcher changes: {modified}")
    if git(source, "status", "--porcelain"):
        raise RuntimeError("Source worktree changed during materialization")
    return {
        "source_revision": revision,
        "source": str(source),
        "checkout": str(destination),
        "patch_sha256": digest(PATCH) if patched else None,
        "source_helpers": original,
        "materialized_helpers": {name: digest(destination / name) for name in original},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--unpatched", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(args.source, args.destination, patched=not args.unpatched)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Materialized launcher: {args.destination}")


if __name__ == "__main__":
    main()
