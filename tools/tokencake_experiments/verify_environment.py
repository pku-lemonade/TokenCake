# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify runtime provenance without launching a timed benchmark case."""

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

from tools.tokencake_experiments.provenance import file_identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument(
        "--role", choices=("target", "source", "baseline"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    # Use the requested checkout even when another editable vLLM is installed.
    sys.path.insert(0, str(checkout))
    import torch
    import vllm._C

    import vllm

    if Path(vllm.__file__).resolve().parent != checkout / "vllm":
        raise ValueError(f"Imported a different checkout: {vllm.__file__}")
    expected_torch = "2.6.0+cu124" if args.role == "source" else "2.11.0+cu130"
    if torch.__version__ != expected_torch:
        raise ValueError(f"Unexpected Torch version: {torch.__version__}")
    devices = [
        {
            "name": torch.cuda.get_device_name(i),
            "capability": torch.cuda.get_device_capability(i),
        }
        for i in range(torch.accelerator.device_count())
    ]
    if len(devices) != 2 or any(d["name"] != "NVIDIA A800-SXM4-80GB" for d in devices):
        raise ValueError(f"Unexpected CUDA visibility: {devices}")
    result = {
        "command": [
            sys.executable,
            "-m",
            "tools.tokencake_experiments.verify_environment",
            *sys.argv[1:],
        ],
        "role": args.role,
        "executable": sys.executable,
        "python_version": sys.version,
        "base_prefix": sys.base_prefix,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "vllm_module": vllm.__file__,
        "extension": file_identity(Path(vllm._C.__file__)),
        "devices": devices,
    }
    if args.role == "source":
        from vllm.v1.core.sched.opt_scheduler import OptScheduler

        result["scheduler"] = OptScheduler.__module__
    else:
        import mooncake.engine
        import mooncake.store

        mooncake_version = importlib.metadata.version("mooncake-transfer-engine")
        if mooncake_version != "0.3.8":
            raise ValueError(f"Unexpected Mooncake version: {mooncake_version}")
        result["mooncake"] = {
            "version": mooncake_version,
            "engine": file_identity(Path(mooncake.engine.__file__)),
            "store": file_identity(Path(mooncake.store.__file__)),
        }
    if args.role == "baseline":
        from vllm.engine.arg_utils import EngineArgs

        config = EngineArgs(
            model="/root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct",
            gpu_memory_utilization=0.50,
            max_model_len=32768,
        ).create_engine_config()
        if (
            config.additional_config
            or config.cache_config.kv_offloading_size is not None
            or config.kv_transfer_config is not None
            or config.scheduler_config.scheduler_cls is not None
            or config.scheduler_config.policy != "fcfs"
        ):
            raise ValueError("Baseline enables a non-native scheduling/offload option")
        result["resolved_baseline"] = {
            "additional_config": config.additional_config,
            "scheduler_policy": config.scheduler_config.policy,
            "scheduler_class": config.scheduler_config.scheduler_cls,
            "offload_size": config.cache_config.kv_offloading_size,
            "connector": config.kv_transfer_config,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Verified {args.role} runtime; evidence: {args.output}")


if __name__ == "__main__":
    main()
