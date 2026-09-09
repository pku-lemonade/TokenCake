# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persist a shared measurement budget across BFCL and SWE invocations."""

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path


class Budget:
    def __init__(self, path: Path, manifest_sha256: str, limit_s: float):
        self.path, self.identity, self.limit_s = path, manifest_sha256, limit_s

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+", buffering=1)
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.stream.seek(0)
            records = [json.loads(line) for line in self.stream if line.strip()]
            if not records:
                self.record(
                    {
                        "event": "budget",
                        "manifest_sha256": self.identity,
                        "limit_s": self.limit_s,
                    }
                )
            elif (
                records[0].get("manifest_sha256") != self.identity
                or records[0].get("limit_s") != self.limit_s
            ):
                raise ValueError("Budget ledger belongs to another frozen campaign")
            if records and records[-1]["event"] == "start":
                raise ValueError(
                    "Prior measurement has no terminal budget record; "
                    "reconcile it before resuming"
                )
            self.used_s = sum(
                row["duration_s"] for row in records if row["event"] == "end"
            )
            return self
        except BaseException:
            self.stream.close()
            raise

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.limit_s - self.used_s)

    def record(self, record: dict):
        self.stream.write(json.dumps(record | {"time": time.time()}) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    @contextmanager
    def measurement(self, label: str):
        if self.remaining_s <= 0:
            raise ValueError("Campaign measurement budget exhausted")
        self.record({"event": "start", "label": label})
        started = time.monotonic()
        try:
            yield self.remaining_s
        finally:
            elapsed = time.monotonic() - started
            self.record({"event": "end", "label": label, "duration_s": elapsed})
            self.used_s += elapsed

    def __exit__(self, *_):
        self.stream.close()
