# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fresh local model services for the four real-agent comparison modes."""

import json
import os
import time
from contextlib import ExitStack
from pathlib import Path

import httpx

from tools.tokencake_experiments.campaign import Device, Settings, affinity_command
from tools.tokencake_experiments.runtime import (
    Activity,
    MetricsMonitor,
    Process,
    free_port,
    gpu_devices,
)

from .inputs import ROOT, repository, resource_limit
from .observe import PressureMonitor
from .transport import MODES, write_json


class Service:
    def __init__(
        self,
        platform: Path,
        output: Path,
        mode: str,
        gpu=0,
        port=8060,
        cancel_event=None,
        model=Settings.model,
        settings=None,
    ):
        if mode not in MODES:
            raise ValueError(mode)
        self.platform, self.output, self.mode = platform, output, mode
        self.port, self.gpu = port, gpu
        self.cancel_event = cancel_event
        self.model = model
        self.settings = settings or {}
        self.process = None
        self.metrics = None
        self.observer = None
        self.stack = ExitStack()

    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=False)
        self.port = free_port(self.port)
        (self.platform / "tmp").mkdir(parents=True, exist_ok=True)
        checkout = ROOT
        identity = repository(checkout)
        python = self.settings.get("python", Settings.python)
        command = [
            python,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            str(self.model),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--dtype",
            self.settings.get("dtype", Settings.dtype),
            "--gpu-memory-utilization",
            str(self.settings.get("gpu_memory_utilization", 0.5)),
            "--max-model-len",
            str(self.settings.get("max_model_len", 32768)),
            "--max-num-batched-tokens",
            str(self.settings.get("max_num_batched_tokens", 8192)),
            "--tensor-parallel-size",
            "1",
            "--pipeline-parallel-size",
            "1",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
        ]
        if self.mode != "base":
            settings = {}
            if self.mode == "agent":
                settings["offload"] = {"enabled": False}
            if self.mode == "offload":
                settings["scheduling"] = {"enabled": False}
            command += ["--additional-config", json.dumps({"tokencake": settings})]
        if self.mode in ("offload", "agent_offload"):
            command += [
                "--kv-offloading-size",
                str(self.settings.get("cpu_kv_gib", 100)),
                "--kv-offloading-backend",
                "native",
            ]
        environment = {
            "PATH": str(Path(python).parent) + os.pathsep + os.environ.get("PATH", ""),
            "CUDA_VISIBLE_DEVICES": str(self.gpu),
            "PYTHONPATH": str(checkout),
            "PYTHONDONTWRITEBYTECODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_USE_V1": "1",
            "VLLM_USE_SIMPLE_KV_OFFLOAD": "0",
            "HF_HOME": str(self.platform / "cache/huggingface"),
            "VLLM_CACHE_ROOT": str(self.platform / "cache/vllm"),
            "TRITON_CACHE_DIR": str(self.platform / "cache/triton"),
            "FLASHINFER_WORKSPACE_BASE": str(self.platform / "cache/flashinfer"),
            "TMPDIR": str(self.platform / "tmp"),
        }
        cpus = self.settings.get("cpus")
        command = affinity_command(command, cpus)
        gpu_info = gpu_devices()
        selected = next(
            (item for item in gpu_info if int(item["index"]) == self.gpu), {}
        )
        device = Device(self.gpu, selected.get("uuid", ""), cpus=cpus)
        environment["CUDA_VISIBLE_DEVICES"] = device.uuid or str(self.gpu)
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        write_json(
            self.output / "launch.json",
            {
                "command": command,
                "environment": environment,
                "cwd": str(checkout),
                "repository": identity,
                "gpu_info": gpu_info,
                "selected_gpu_index": self.gpu,
                "selected_gpu_uuid": device.uuid,
                "cpus": cpus,
                "cgroup_memory_max": resource_limit("memory.max"),
            },
        )
        started = time.time()
        try:
            self.process = Process(
                command, environment, checkout, self.output / "server.log", cpus=cpus
            )
            self.stack.callback(self.process.close)
            activity = Activity()
            activity.set(device, self.mode, [self.process.process.pid])
            self.observer = PressureMonitor(
                device, activity, self.output / "resources.jsonl"
            )
            self.stack.enter_context(self.observer)
            write_json(
                self.output / "process.json",
                {"pid": self.process.process.pid, "started_at": started},
            )
            with httpx.Client(trust_env=False, timeout=3) as client:
                while time.time() - started < 900:
                    if self.cancel_event is not None and self.cancel_event.is_set():
                        raise InterruptedError("Parallel service startup cancelled")
                    if self.process.process.poll() is not None:
                        raise RuntimeError(
                            f"Service exited: {self.output / 'server.log'}"
                        )
                    try:
                        response = client.get(f"http://127.0.0.1:{self.port}/v1/models")
                        if response.is_success:
                            try:
                                initial = client.get(
                                    f"http://127.0.0.1:{self.port}/metrics"
                                )
                                initial.raise_for_status()
                                (self.output / "metrics-start.prom").write_text(
                                    initial.text
                                )
                            except httpx.HTTPError as exc:
                                (self.output / "metrics-start.error.txt").write_text(
                                    str(exc)
                                )
                            write_json(
                                self.output / "ready.json",
                                {
                                    "ready_at": time.time(),
                                    "initialization_s": time.time() - started,
                                    "models": response.json(),
                                },
                            )
                            self.metrics = MetricsMonitor(
                                self.port, self.output / "metrics.jsonl"
                            )
                            self.stack.enter_context(self.metrics)
                            return self
                    except httpx.HTTPError:
                        pass
                    time.sleep(1)
            raise TimeoutError("Model service did not become ready within 900 seconds")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        try:
            if self.process is not None:
                with httpx.Client(trust_env=False, timeout=5) as client:
                    try:
                        response = client.get(f"http://127.0.0.1:{self.port}/metrics")
                        if response.is_success:
                            (self.output / "metrics.prom").write_text(response.text)
                    except httpx.HTTPError:
                        pass
        finally:
            try:
                self.stack.close()
            finally:
                if self.metrics is not None:
                    write_json(
                        self.output / "metrics-monitor.json", self.metrics.summary
                    )
                if self.observer is not None:
                    write_json(
                        self.output / "resources-monitor.json", self.observer.summary
                    )
                if self.process is not None:
                    write_json(
                        self.output / "stopped.json",
                        {
                            "stopped_at": time.time(),
                            "exit_code": self.process.process.returncode,
                        },
                    )


def worker_python(platform: Path, *, grading=False):
    candidates = [platform / "environments/grading/.venv/bin/python"] if grading else []
    candidates.append(platform / ".venv/bin/python")
    return next((str(path) for path in candidates if path.is_file()), Settings.python)


def worker_environment(platform: Path, task_output: Path) -> dict[str, str]:
    # Runtime flags and credentials from the parent shell must not alter agent
    # behavior or be exposed to benchmark shell commands.
    keys = ("PATH", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR")
    environment = {key: os.environ[key] for key in keys if key in os.environ}
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                [
                    str(ROOT),
                    str(
                        platform / "sources/gorilla/berkeley-function-call-leaderboard"
                    ),
                ]
            ),
            "BFCL_PROJECT_ROOT": str(task_output / "bfcl-state"),
            "MSWEA_GLOBAL_CONFIG_DIR": str(task_output / "mini-config"),
            "MSWEA_SILENT_STARTUP": "1",
            "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": "1",
            "LITELLM_LOCAL_MODEL_COST_MAP": "true",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HOME": str(platform / "cache/huggingface"),
            "TMPDIR": str(platform / "tmp"),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    return environment
