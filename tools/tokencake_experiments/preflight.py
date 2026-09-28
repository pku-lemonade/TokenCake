# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare experiments from the current checkout and selected local resources."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

from tools.tokencake_experiments.campaign import (
    PACKAGE,
    ROOT,
    Case,
    Device,
    Settings,
    add_settings_arguments,
    initial_queues,
    server_command,
    settings_from_args,
    workload_parameters,
)
from tools.tokencake_experiments.dataset import load_dataset
from tools.tokencake_experiments.materialize import (
    WORKLOAD_PROFILES,
    digest,
    materialize,
)
from tools.tokencake_experiments.provenance import capture
from tools.tokencake_experiments.report import Identity, content_hash
from tools.tokencake_experiments.runtime import (
    gpu_devices,
    write_json,
)


def inherited_environment():
    prefixes = (
        "VLLM_",
        "TOKENCAKE_",
        "MOONCAKE_",
        "CUDA_",
        "NCCL_",
        "TORCH_",
        "HF_",
        "TRITON_",
        "OMP_",
    )
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if (key.startswith(prefixes) or key in ("PATH", "LD_LIBRARY_PATH"))
        and not key.endswith(("_KEY", "_TOKEN", "_PASSWORD"))
    }


def adapter(command, checkout, parameters, output, *, python=sys.executable):
    parameters_path = output.with_suffix(".input.json")
    write_json(parameters_path, parameters)
    with output.with_suffix(".log").open("x") as log:
        subprocess.run(
            [
                str(python),
                str(PACKAGE / "source_adapter.py"),
                command,
                "--checkout",
                str(checkout),
                "--input",
                str(parameters_path),
                "--output",
                str(output),
            ],
            cwd=ROOT,
            env=os.environ | {"PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=300,
        )


def hardware_info(settings):
    """Record available GPU information without making it a launch prerequisite."""
    observed = gpu_devices()
    lookup = {int(row["index"]): row for row in observed}
    return {
        "devices": [
            asdict(
                Device(index, lookup.get(index, {}).get("uuid", ""), cpus=settings.cpus)
            )
            for index in settings.gpus
        ],
        "observed_gpus": observed,
    }


def carry_exclusions(previous: Path, identities: list[Identity]) -> list[dict]:
    lookup = {identity.case.name: identity for identity in identities}
    results = []
    inherited = previous / "prior-exclusions.json"
    paths = sorted(previous.glob("cases/**/result.json"))
    rows = json.loads(inherited.read_text()) if inherited.exists() else []
    rows.extend(json.loads(path.read_text()) for path in paths)
    for row in rows:
        if row.get("qualifying"):
            continue
        case = Case(**row["identity"]["case"])
        target = lookup.get(case.name)
        if target is None:
            continue
        carried = row | {"budget_identity": target.key}
        results.append(carried)
    return results


def prepare(
    run_root,
    *,
    prior_exclusions=None,
    snapshot_target=False,
    cases=None,
    workload_profile="conversation-tools",
    workload_dataset=None,
    settings=None,
    mooncake_config=None,
    mooncake_master=None,
):
    settings = settings or Settings()
    if workload_profile not in WORKLOAD_PROFILES:
        raise ValueError(f"Unknown workload profile: {workload_profile}")
    workload_dataset = Path(
        workload_dataset or PACKAGE / "datasets" / f"{workload_profile}.json"
    ).resolve()
    dataset = load_dataset(workload_dataset)
    cases = (
        cases
        if cases is not None
        else [case for queue in initial_queues(settings)["primary"] for case in queue]
    )
    for case in cases:
        settings.device(case)
        if case.mode == "old-offload-agent":
            raise ValueError(
                "New campaigns use only implementations in the current repository"
            )
    hardware = hardware_info(settings)
    mooncake = (
        json.loads(Path(mooncake_config).read_text()) if mooncake_config else None
    )
    master = (
        {
            "path": str(
                mooncake_master or Path(settings.python).parent / "mooncake_master"
            )
        }
        if mooncake is not None
        else None
    )
    run_root = Path(run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    write_json(run_root / "hardware.json", hardware)
    provenance = capture(
        SimpleNamespace(
            target=ROOT,
            model=settings.model,
            dataset=workload_dataset,
            python=settings.python,
        )
    )
    settings = replace(settings, model=provenance["model"])
    write_json(run_root / "provenance.json", provenance)
    target_snapshot = run_root / "runtime" if snapshot_target else None
    if target_snapshot:
        shutil.copytree(
            ROOT / "vllm",
            target_snapshot / "vllm",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    target = {
        "checkout": str(target_snapshot or ROOT),
        "snapshot": bool(target_snapshot),
    }
    code = {"target": target, "baseline": dict(target)}
    write_json(run_root / "code.json", code)
    launcher = materialize(
        ROOT,
        run_root / "launcher",
        workload_profile=workload_profile,
        workload_dataset=workload_dataset,
    )
    write_json(run_root / "launcher.json", launcher)
    parameters = workload_parameters(settings) | {
        "dataset": str(workload_dataset),
        "num_requests": len(dataset.applications),
        "task": dataset.name,
        "seed": dataset.provenance.get("seed", 42),
        "workload_dataset_sha256": launcher["workload_dataset_sha256"],
        "qps": sorted({case.qps for case in cases}, reverse=True),
    }
    if dataset.input_composition is not None:
        parameters["input_composition"] = dataset.input_composition
    if dataset.tool_instruction_max_tokens is not None:
        parameters["tool_instruction_max_tokens"] = dataset.tool_instruction_max_tokens
    workload = dataset.freeze(parameters["qps"])
    write_json(run_root / "workload.json", workload)
    for qps, offsets in workload["arrivals"].items():
        write_json(run_root / f"arrivals-{qps}.json", {"offsets_s": offsets})
    environment = content_hash({"python": settings.python})
    inputs = content_hash(
        {
            "parameters": parameters,
            "workload": workload,
            "model": provenance["model"],
            "dataset": provenance["dataset"],
        }
    )
    inherited = inherited_environment()
    devices = {device["index"]: Device(**device) for device in hardware["devices"]}
    identities = [
        Identity(
            case,
            "current-checkout",
            environment,
            content_hash(
                {
                    "helpers": launcher["materialized_helpers"],
                    "wrapper": digest(PACKAGE / "launch_client.py"),
                }
            ),
            inputs,
            content_hash(
                {
                    "server": server_command(
                        case,
                        0,
                        target_checkout=Path(code["target"]["checkout"]),
                        settings=settings,
                        device=devices[settings.device(case).index],
                    )[:2],
                    "inherited_environment": inherited,
                    "mooncake": mooncake if case.mode == "mooncake" else None,
                }
            ),
        )
        for case in cases
    ]
    if prior_exclusions is not None:
        write_json(
            run_root / "prior-exclusions.json",
            carry_exclusions(Path(prior_exclusions).resolve(), identities),
        )
    frozen = {
        "schema_version": 2,
        "settings": settings.payload(),
        "devices": hardware["devices"],
        "workload_profile": workload_profile,
        "parameters": parameters,
        "code": code,
        "workload_sha256": workload["workload_sha256"],
        "arrivals": workload["arrivals"],
        "identities": [identity.payload() for identity in identities],
        "mooncake": mooncake,
        "mooncake_master": master,
    }
    write_json(run_root / "frozen.json", frozen)
    return frozen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    add_settings_arguments(parser)
    args = parser.parse_args()
    print(
        json.dumps(prepare(args.run_root, settings=settings_from_args(args)), indent=2)
    )
