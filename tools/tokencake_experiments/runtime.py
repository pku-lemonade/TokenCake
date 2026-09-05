# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fresh-process execution and diagnostic GPU/process observation."""

import csv
import io
import json
import os
import signal
import socket
import subprocess
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx
import psutil
from prometheus_client.parser import text_string_to_metric_families

from tools.tokencake_experiments.campaign import Device


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def query_nvidia(kind: str, fields: list[str]) -> list[dict[str, str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-{kind}=" + ",".join(fields),
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return [
        dict(zip(fields, (value.strip() for value in row), strict=True))
        for row in csv.reader(io.StringIO(result.stdout))
        if row
    ]


def gpu_devices() -> list[dict[str, str]]:
    return query_nvidia("gpu", ["index", "uuid", "name", "memory.total"])


def free_port(preferred: int) -> int:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", preferred))
        except OSError:
            sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def cpu_set(value: str) -> set[int]:
    result = set()
    for item in value.split(","):
        bounds = [int(part) for part in item.split("-")]
        result.update(range(bounds[0], bounds[-1] + 1))
    return result


class Process:
    def __init__(
        self,
        command: list[str],
        environment: dict[str, str],
        cwd: Path,
        log: Path,
        *,
        cpus: str,
    ):
        self.log = log.open("x")
        try:
            self.process = subprocess.Popen(
                command,
                cwd=cwd,
                env=os.environ | environment,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            with suppress(psutil.NoSuchProcess):
                psutil.Process(self.process.pid).cpu_affinity(sorted(cpu_set(cpus)))
        except BaseException:
            if hasattr(self, "process"):
                self.close()
            self.log.close()
            raise

    def close(self) -> None:
        with suppress(ProcessLookupError):
            os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=30)
        finally:
            # The group may outlive its launcher process.
            with suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)
            self.log.close()


class Activity:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.cases: dict[str, dict] = {}

    def set(self, device: Device, case: str, roots: list[int]) -> None:
        with self.lock:
            self.cases[device.uuid] = {"case": case, "roots": roots.copy()}

    def clear(self, device: Device) -> None:
        with self.lock:
            self.cases.pop(device.uuid, None)

    def snapshot(self) -> dict:
        with self.lock:
            return {uuid: entry.copy() for uuid, entry in self.cases.items()}


class Monitor:
    def __init__(self, device: Device, activity: Activity, output: Path) -> None:
        self.device, self.activity, self.output = device, activity, output
        self.stop = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name=f"gpu-{device.index}-monitor"
        )
        self.summary = {
            "external_gpu_process_detected": False,
            "monitor_query_failed": False,
            "affinity_violation": False,
            "thermal_validity_enforced": False,
            "samples": 0,
        }
        self.owned: set[int] = set()

    def sample(self) -> dict:
        active = self.activity.snapshot()
        own = active.get(self.device.uuid, {})
        self.owned.update(own.get("roots", []))
        affinity = []
        for root in own.get("roots", []):
            try:
                process = psutil.Process(root)
                descendants = [process, *process.children(recursive=True)]
            except psutil.NoSuchProcess:
                continue
            for process in descendants:
                try:
                    cpus = process.cpu_affinity()
                    affinity.append({"pid": process.pid, "cpus": cpus})
                    self.owned.add(process.pid)
                    if not set(cpus) <= cpu_set(self.device.cpus):
                        self.summary["affinity_violation"] = True
                except psutil.NoSuchProcess:
                    pass
        processes = query_nvidia(
            "compute-apps", ["gpu_uuid", "pid", "process_name", "used_gpu_memory"]
        )
        foreign = [
            process
            for process in processes
            if process["gpu_uuid"] == self.device.uuid
            and int(process["pid"]) not in self.owned
        ]
        self.summary["external_gpu_process_detected"] |= bool(foreign)
        thermal = query_nvidia(
            "gpu",
            [
                "uuid",
                "temperature.gpu",
                "power.draw",
                "power.limit",
                "clocks.sm",
                "clocks.mem",
            ],
        )
        self.summary["samples"] += 1
        return {
            "timestamp": time.time(),
            "monotonic": time.monotonic(),
            "activity": active,
            "processes": processes,
            "foreign": foreign,
            "affinity": affinity,
            "thermal": thermal,
        }

    def _run(self) -> None:
        with self.output.open("x") as stream:
            while not self.stop.is_set():
                try:
                    sample = self.sample()
                except Exception as exc:
                    self.summary["monitor_query_failed"] = True
                    sample = {
                        "timestamp": time.time(),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                stream.write(json.dumps(sample, sort_keys=True) + "\n")
                stream.flush()
                self.stop.wait(2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError("GPU monitor did not stop")


def wait_ready(
    server: Process,
    port: int,
    timeout: float = 600,
    *,
    stop: threading.Event | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    with httpx.Client(timeout=5) as client:
        while time.monotonic() < deadline:
            if stop is not None and stop.is_set():
                raise InterruptedError("Campaign interrupted")
            if server.process.poll() is not None:
                raise RuntimeError(
                    f"Server exited with code {server.process.returncode}"
                )
            try:
                if client.get(f"http://127.0.0.1:{port}/health").status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
    raise TimeoutError("Server readiness timed out")


def metric_values(text: str) -> list[dict]:
    return [
        {"name": sample.name, "labels": sample.labels, "value": sample.value}
        for family in text_string_to_metric_families(text)
        for sample in family.samples
    ]


def metric_sum(samples: list[dict], name: str, **labels: str) -> float:
    return sum(
        sample["value"]
        for sample in samples
        if sample["name"] == name
        and all(sample["labels"].get(key) == value for key, value in labels.items())
    )
