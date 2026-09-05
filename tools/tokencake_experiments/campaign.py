# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The accepted A800 matrix and version-specific serving commands."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "vllm_agent"
MODEL = Path("/root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct")
WORKLOAD_REVISION = "be5f23ca50818f74f485ea82f42138ec9506e1f0"
BASELINE_REVISION = "0b3ba88f165976e77ca5e6a7a3f5bba4562b80af"
QPS = (1.0, 0.5, 0.1)
MOONCAKE_SETTINGS = {
    "global_segment_size": 100 * 2**30,
    "local_buffer_size": 2**30,
    "protocol": "tcp",
}
Mode = Literal["native", "agent", "offload-agent", "old-offload-agent", "mooncake"]


@dataclass(frozen=True)
class Device:
    index: int
    uuid: str
    numa: int
    cpus: str


DEVICES = (
    Device(0, "GPU-dfe6761a-23f4-c34c-9c03-01da02020e5f", 0, "0-35,72-107"),
    Device(1, "GPU-ce81ab13-d8a0-a49d-c908-f876b8eb2087", 1, "36-71,108-143"),
)


@dataclass(frozen=True)
class Case:
    mode: Mode
    qps: float
    phase: str = "phase-1"

    def __post_init__(self) -> None:
        if self.mode not in (
            "native",
            "agent",
            "offload-agent",
            "old-offload-agent",
            "mooncake",
        ):
            raise ValueError("Unknown benchmark mode")
        if self.qps not in QPS:
            raise ValueError("The accepted QPS values are 1.0, 0.5 and 0.1")
        if self.phase not in ("phase-1", "phase-2"):
            raise ValueError("Unknown implementation phase")
        if self.phase == "phase-2" and self.mode != "offload-agent":
            raise ValueError("Only target offload-agent has an implementation phase")

    @property
    def device(self) -> Device:
        return DEVICES[int(self.mode in ("agent", "mooncake"))]

    @property
    def name(self) -> str:
        return f"{self.phase}/{self.mode}/qps-{self.qps:.1f}"

    def payload(self) -> dict:
        return asdict(self) | {"device": asdict(self.device), "name": self.name}


def initial_queues() -> dict[str, list[list[Case]]]:
    return {
        "primary": [
            [Case(mode, qps) for mode in ("native", "offload-agent") for qps in QPS],
            [Case("agent", qps) for qps in QPS],
        ],
        "references": [
            [Case("old-offload-agent", qps) for qps in QPS],
            [Case("mooncake", qps) for qps in QPS],
        ],
    }


def workload_parameters() -> dict:
    return {
        "model": str(MODEL),
        "dataset": str(SOURCE / "dataset/agentcodeclean_new.json"),
        "workload_revision": WORKLOAD_REVISION,
        "seed": 42,
        "num_requests": 24,
        "qps": list(QPS),
        "gpu_memory_utilization": 0.5,
        "max_model_len": 32768,
        "cpu_offload_gib": 100,
    }


def server_command(case: Case, port: int) -> tuple[list[str], dict[str, str], Path]:
    old = case.mode == "old-offload-agent"
    python = ROOT / (".venv/source/.venv/bin/python" if old else ".venv/bin/python")
    checkout = (
        SOURCE if old else ROOT / ".venv/baseline" if case.mode == "native" else ROOT
    )
    command = ["taskset", "-c", case.device.cpus, str(python), "-m"]
    if old:
        command += ["vllm.entrypoints.openai.api_server", "--model", str(MODEL)]
    else:
        command += ["vllm.entrypoints.cli.main", "serve", str(MODEL)]
    command += [
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--gpu-memory-utilization",
        "0.5",
        "--max-model-len",
        "32768",
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "1",
    ]
    if case.mode == "agent":
        command += [
            "--additional-config",
            json.dumps({"tokencake": {"offload": {"enabled": False}}}),
        ]
    elif case.mode == "offload-agent":
        command += [
            "--additional-config",
            '{"tokencake":{}}',
            "--kv-offloading-size",
            "100",
            "--kv-offloading-backend",
            "native",
        ]
    elif old:
        command += [
            "--swap-space",
            "100",
            "--scheduling-policy",
            "agent",
            "--scheduler-cls",
            "vllm.v1.core.sched.opt_scheduler.OptScheduler",
            "--enable-agent-scheduling",
            "--enable-kvcache-cpu-offloading",
        ]
    elif case.mode == "mooncake":
        command += [
            "--kv-transfer-config",
            json.dumps(
                {"kv_connector": "MooncakeStoreConnector", "kv_role": "kv_both"}
            ),
        ]
    environment = {
        "CUDA_VISIBLE_DEVICES": str(case.device.index),
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_USE_V1": "1",
        "PYTHONPATH": str(checkout),
        "PYTHONDONTWRITEBYTECODE": "1",
        "VLLM_SERVER_DEV_MODE": "1",
    }
    if not old:
        environment["VLLM_USE_SIMPLE_KV_OFFLOAD"] = "0"
    return command, environment, checkout


def client_command(
    case: Case, port: int, case_dir: Path, checkout: Path, arrival_trace: Path
) -> tuple[list[str], dict[str, str], Path]:
    old = case.mode == "old-offload-agent"
    python = ROOT / (".venv/source/.venv/bin/python" if old else ".venv/bin/python")
    command = ["taskset", "-c", case.device.cpus, str(python)]
    if old:
        command += [str(SOURCE / "vllm_serving.py")]
        checkout = SOURCE
    else:
        command += [str(PACKAGE / "launch_client.py"), str(checkout)]
    command += [
        "--port",
        str(port),
        "--model_path",
        str(MODEL),
        "--dataset",
        str(SOURCE / "dataset/agentcodeclean_new.json"),
        "--workload_source_revision",
        WORKLOAD_REVISION,
        "--task",
        "code-paper-pressure",
        "--request_rate",
        str(case.qps),
        "--num_requests",
        "24",
        "--seed",
        "42",
        "--output_dir",
        str(case_dir / "app_results"),
        "--output_file",
        str(case_dir / "output_record.json"),
        "--arrival_trace_file",
        str(arrival_trace),
    ]
    if case.mode in ("native", "agent", "offload-agent"):
        command += ["--tokencake-mode", case.mode]
    elif case.mode == "mooncake":
        command += ["--disable_mcp_notifications"]
    return (
        command,
        {
            "CUDA_VISIBLE_DEVICES": str(case.device.index),
            "PYTHONPATH": str(SOURCE if old else ROOT),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        checkout,
    )
