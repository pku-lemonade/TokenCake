# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable experiment settings and serving commands for the current checkout."""

import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
QPS = (1.0, 0.5, 0.1)
COMPONENT_QPS = (1.0, 0.5, 0.2, 0.1, 0.05)
CURRENT_MODES = ("native", "agent", "offload", "offload-agent", "mooncake")
# Historical result identities remain readable; new runs use CURRENT_MODES.
Mode = Literal[
    "native", "agent", "offload", "offload-agent", "old-offload-agent", "mooncake"
]


@dataclass(frozen=True)
class Device:
    index: int
    uuid: str = ""
    numa: int | None = None
    cpus: str | None = None


@dataclass(frozen=True)
class Settings:
    model: str = "Qwen/Qwen2.5-14B-Instruct"
    gpus: tuple[int, ...] = (0,)
    cpus: str | None = None
    python: str = sys.executable
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.5
    max_model_len: int = 32768
    cpu_offload_gib: float = 100

    def __post_init__(self):
        executable = shutil.which(self.python) or self.python
        # Preserve the venv symlink while making snapshot working directories safe.
        object.__setattr__(
            self, "python", str(Path(executable).expanduser().absolute())
        )
        if not self.model or not self.gpus or len(set(self.gpus)) != len(self.gpus):
            raise ValueError("Specify a model and distinct GPU indices")
        if any(index < 0 for index in self.gpus):
            raise ValueError("GPU indices must be nonnegative")
        if not 0 < self.gpu_memory_utilization <= 1 or self.max_model_len <= 0:
            raise ValueError("Invalid GPU memory utilization or maximum model length")
        if not math.isfinite(self.cpu_offload_gib) or self.cpu_offload_gib <= 0:
            raise ValueError("CPU offload capacity must be positive and finite")

    def device(self, case):
        index = self.gpus[0] if case.gpu_index is None else case.gpu_index
        if index not in self.gpus:
            raise ValueError(f"GPU {index} is not selected by this experiment")
        return Device(index, cpus=self.cpus)

    def payload(self):
        return asdict(self)


def add_settings_arguments(parser):
    parser.add_argument(
        "--model",
        default=Settings.model,
        help="Local model directory or cached Hugging Face model ID",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=[0],
        help="Physical nvidia-smi GPU indices; one GPU is sufficient",
    )
    parser.add_argument("--cpus", help="Optional CPU affinity, e.g. 0-7,16-23")
    parser.add_argument(
        "--python", default=sys.executable, help="Serving and client interpreter"
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "half", "float16", "bfloat16", "float", "float32"),
    )
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--cpu-offload-gib", type=float, default=100)


def settings_from_args(args):
    return Settings(
        **{
            name: tuple(args.gpus) if name == "gpus" else getattr(args, name)
            for name in Settings.__dataclass_fields__
        }
    )


def affinity_command(command, cpus):
    return ["taskset", "-c", cpus, *command] if cpus else command


@dataclass(frozen=True)
class Case:
    mode: Mode
    qps: float
    phase: str = "phase-1"
    gpu_index: int | None = None

    def __post_init__(self):
        if self.gpu_index is not None and self.gpu_index < 0:
            raise ValueError("GPU index must be nonnegative")
        if self.mode not in (*CURRENT_MODES, "old-offload-agent"):
            raise ValueError("Unknown benchmark mode")
        if not math.isfinite(self.qps) or self.qps <= 0:
            raise ValueError("QPS must be positive and finite")
        if self.phase not in ("phase-1", "phase-2"):
            raise ValueError("Unknown implementation phase")
        if self.phase == "phase-2" and self.mode != "offload-agent":
            raise ValueError("Only target offload-agent has an implementation phase")

    @property
    def name(self):
        suffix = "" if self.gpu_index is None else f"/gpu-{self.gpu_index}"
        return f"{self.phase}/{self.mode}/qps-{float(self.qps)}{suffix}"

    def payload(self):
        return asdict(self) | {"name": self.name}


def group_cases(cases, settings):
    queues = {gpu: [] for gpu in settings.gpus}
    for case in cases:
        queues[settings.device(case).index].append(case)
    return [queue for queue in queues.values() if queue]


def make_cases(settings, modes, qps_values):
    # Keep a mode on the same GPU at every offered load.
    return [
        Case(mode, qps, gpu_index=settings.gpus[index % len(settings.gpus)])
        for index, mode in enumerate(modes)
        for qps in qps_values
    ]


def initial_queues(settings=None):
    settings = settings or Settings()
    return {
        "primary": group_cases(
            make_cases(settings, ("native", "agent", "offload-agent"), QPS), settings
        )
    }


def component_queues(settings=None):
    settings = settings or Settings()
    return group_cases(
        make_cases(
            settings, ("native", "agent", "offload", "offload-agent"), COMPONENT_QPS
        ),
        settings,
    )


def workload_parameters(settings=None):
    settings = settings or Settings()
    return {
        "model": settings.model,
        "seed": 42,
        "num_requests": 24,
        "qps": list(QPS),
        "gpu_memory_utilization": settings.gpu_memory_utilization,
        "max_model_len": settings.max_model_len,
        "cpu_offload_gib": settings.cpu_offload_gib,
    }


def server_command(case, port, *, target_checkout=ROOT, settings=None, device=None):
    settings = settings or Settings()
    device = device or settings.device(case)
    if case.mode not in CURRENT_MODES:
        raise ValueError(
            "Historical implementations cannot be launched; use the current checkout"
        )
    command = [
        settings.python,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        settings.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        settings.dtype,
        "--gpu-memory-utilization",
        str(settings.gpu_memory_utilization),
        "--max-model-len",
        str(settings.max_model_len),
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "1",
    ]
    if case.mode in ("agent", "offload", "offload-agent"):
        config = {}
        if case.mode == "agent":
            config["offload"] = {"enabled": False}
        elif case.mode == "offload":
            config["scheduling"] = {"enabled": False}
        command += ["--additional-config", json.dumps({"tokencake": config})]
    if case.mode in ("offload", "offload-agent"):
        command += [
            "--kv-offloading-size",
            str(settings.cpu_offload_gib),
            "--kv-offloading-backend",
            "native",
        ]
    if case.mode == "mooncake":
        command += [
            "--kv-transfer-config",
            json.dumps(
                {"kv_connector": "MooncakeStoreConnector", "kv_role": "kv_both"}
            ),
        ]
    environment = {
        "CUDA_VISIBLE_DEVICES": device.uuid or str(device.index),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_USE_V1": "1",
        "PYTHONPATH": str(target_checkout),
        "PYTHONDONTWRITEBYTECODE": "1",
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_USE_SIMPLE_KV_OFFLOAD": "0",
    }
    return affinity_command(command, device.cpus), environment, target_checkout


def client_command(
    case,
    port,
    case_dir,
    checkout,
    arrival_trace,
    *,
    settings=None,
    device=None,
    workload=None,
):
    settings = settings or Settings()
    device = device or settings.device(case)
    workload = workload or workload_parameters(settings)
    command = [
        settings.python,
        str(PACKAGE / "launch_client.py"),
        str(checkout),
        "--port",
        str(port),
        "--model_path",
        settings.model,
        "--dataset",
        str(checkout / "workload-dataset.json"),
        "--task",
        workload.get("task", "code-paper-pressure"),
        "--request_rate",
        str(case.qps),
        "--num_requests",
        str(workload["num_requests"]),
        "--seed",
        str(workload["seed"]),
        "--output_dir",
        str(case_dir / "app_results"),
        "--output_file",
        str(case_dir / "output_record.json"),
        "--arrival_trace_file",
        str(arrival_trace),
        "--tokencake-mode",
        "native" if case.mode == "mooncake" else case.mode,
    ]
    return (
        affinity_command(command, device.cpus),
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": str(ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        checkout,
    )
