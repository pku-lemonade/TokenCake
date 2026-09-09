# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One application per process, with durable terminal records and cleanup."""

import argparse
import ctypes
import json
import time
import traceback
from contextlib import suppress
from pathlib import Path

import psutil

from .inputs import MODEL
from .transport import Journal, TaskContext, Transport, write_json


def cleanup_children():
    children = psutil.Process().children(recursive=True)
    for child in reversed(children):
        with suppress(psutil.NoSuchProcess):
            child.terminate()
    _, alive = psutil.wait_procs(children, timeout=3)
    for child in alive:
        with suppress(psutil.NoSuchProcess):
            child.kill()
    psutil.wait_procs(alive, timeout=3)


def run(spec: dict, output: Path):
    # Reparent orphaned shell grandchildren here, so final cleanup also owns
    # background processes that outlived the immediate shell.
    if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise RuntimeError("Could not enable benchmark child-process cleanup")
    journal = Journal(output / "journal.jsonl")
    context = TaskContext(**spec["context"])
    transport = Transport(spec["base_url"], context, journal)
    actual_start = time.time()
    journal.record("task_start", context=spec["context"], actual_start=actual_start)
    result = {}
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
        if spec["benchmark"] == "bfcl":
            from .bfcl import run as run_bfcl

            prediction = run_bfcl(spec["task"], transport, str(MODEL), tokenizer)
            result = {"status": "completed", "prediction": prediction}
        else:
            from .mini import run as run_mini

            prediction = run_mini(
                spec["task"]["problem_statement"],
                transport,
                str(MODEL),
                tokenizer,
                Path(spec["preset"]),
                Path(spec["task_directory"]) / "repo",
                spec["task_environment"],
                output / "trajectory.json",
            )
            (output / "prediction.patch").write_text(prediction.get("submission", ""))
            result = {
                "status": prediction.get("exit_status", "unknown"),
                "prediction": prediction,
            }
    except Exception as exc:
        result = {
            "status": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        ended = time.time()
        if not result:
            result = {"status": "interrupted"}
        result.update(
            {
                "task_id": context.task_id,
                "mode": context.mode,
                "arrived_at": context.arrived_at,
                "actual_start": actual_start,
                "ended_at": ended,
                "e2e_s": ended - context.arrived_at,
                "client_queue_s": actual_start - context.arrived_at,
                "model_calls": transport.calls,
            }
        )
        try:
            write_json(output / "result.json", result)
            journal.record("task_end", result=result)
        finally:
            cleanup_error = None
            try:
                try:
                    transport.close()
                finally:
                    cleanup_children()
            except Exception as exc:
                cleanup_error = str(exc)
                raise
            finally:
                write_json(
                    output / "cleanup.json",
                    {"ended_at": time.time(), "error": cleanup_error},
                )
                journal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.spec.read_text()), args.output)


if __name__ == "__main__":
    main()
