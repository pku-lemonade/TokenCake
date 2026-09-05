# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify frozen inputs and prepare isolated launchers before timed cases."""

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import psutil

from tools.tokencake_experiments.campaign import (
    BASELINE_REVISION,
    DEVICES,
    MOONCAKE_SETTINGS,
    PACKAGE,
    ROOT,
    SOURCE,
    Case,
    initial_queues,
    server_command,
    workload_parameters,
)
from tools.tokencake_experiments.materialize import (
    SOURCE_REVISION,
    digest,
    git,
    materialize,
)
from tools.tokencake_experiments.provenance import capture
from tools.tokencake_experiments.report import Identity, content_hash, measurements
from tools.tokencake_experiments.runtime import (
    cpu_set,
    free_port,
    gpu_devices,
    query_nvidia,
    write_json,
)

MOONCAKE_CONFIGURATION = ROOT.parent / "Mooncake-TokenCake"
MOONCAKE_REVISION = "696c9a14f30ffeacda1707e8f712b7a214460be6"


def inherited_environment() -> dict[str, str]:
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


def adapter(
    command: str,
    checkout: Path,
    parameters: dict,
    output: Path,
    *,
    source: bool = False,
) -> None:
    parameters_path = output.with_suffix(".input.json")
    write_json(parameters_path, parameters)
    python = ROOT / (".venv/source/.venv/bin/python" if source else ".venv/bin/python")
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
            cwd=SOURCE if source else ROOT,
            env=os.environ
            | {
                "PYTHONPATH": str(SOURCE if source else ROOT),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=300,
        )


def hardware_preflight() -> dict:
    observed = gpu_devices()
    lookup = {row["uuid"]: row for row in observed}
    for device in DEVICES:
        row = lookup.get(device.uuid)
        if (
            row is None
            or int(row["index"]) != device.index
            or row["name"] != "NVIDIA A800-SXM4-80GB"
        ):
            raise RuntimeError(f"Frozen GPU assignment changed: {observed}")
        numa = (
            Path(f"/sys/devices/system/node/node{device.numa}/cpulist")
            .read_text()
            .strip()
        )
        if cpu_set(numa) != cpu_set(device.cpus):
            raise RuntimeError(f"Frozen NUMA CPU set changed: {numa}")
        if not cpu_set(device.cpus) <= set(psutil.Process().cpu_affinity()):
            raise RuntimeError("Driver affinity excludes an assigned NUMA CPU set")
    processes = query_nvidia("compute-apps", ["gpu_uuid", "pid", "process_name"])
    if processes:
        raise RuntimeError(f"GPUs are in use before the campaign: {processes}")
    for executable in ("taskset", "uv", str(ROOT / ".venv/bin/mooncake_master")):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Missing executable: {executable}")
    available = psutil.virtual_memory().available
    if available < 264 * 2**30:
        raise RuntimeError(
            "Insufficient host memory for the declared concurrent offload servers"
        )
    return {
        "gpus": observed,
        "devices": [asdict(d) for d in DEVICES],
        "available_host_bytes": available,
        "available_ports": [free_port(8055), free_port(8056)],
        "external_processes": processes,
    }


def verify_code() -> dict[str, dict]:
    configurations = {
        "target": (ROOT, None),
        "source": (SOURCE, SOURCE_REVISION),
        "baseline": (ROOT / ".venv/baseline", BASELINE_REVISION),
        "mooncake_configuration": (MOONCAKE_CONFIGURATION, MOONCAKE_REVISION),
    }
    result = {}
    for name, (checkout, expected) in configurations.items():
        revision = git(checkout, "rev-parse", "HEAD")
        if expected is not None and revision != expected:
            raise RuntimeError(f"Frozen {name} revision changed")
        status = (
            git(checkout, "status", "--porcelain", "--", "vllm")
            if name == "target"
            else git(checkout, "status", "--porcelain")
        )
        if status:
            raise RuntimeError(f"Uncommitted executable changes in {name}: {status}")
        result[name] = {"commit": revision, "checkout": str(checkout)}
        if name != "mooncake_configuration":
            result[name]["runtime_tree"] = git(checkout, "rev-parse", "HEAD:vllm")
    return result


def carry_exclusions(previous: Path, identities: list[Identity]) -> list[dict]:
    lookup = {identity.case.name: identity for identity in identities}
    results = []
    inherited = previous / "prior-exclusions.json"
    paths = sorted(previous.glob("cases/**/result.json"))
    rows = json.loads(inherited.read_text()) if inherited.exists() else []
    rows.extend(json.loads(path.read_text()) for path in paths)
    if len(paths) != len(list(previous.glob("cases/**/attempt.json"))):
        raise RuntimeError("Prior campaign has unfinished attempts")
    for row in rows:
        if row.get("qualifying"):
            raise ValueError("This recovery only transfers excluded launch charges")
        case = Case(**row["identity"]["case"])
        target = lookup[case.name]
        carried = row | {"budget_identity": target.key}
        results.append(carried)
    for identity in identities:
        measurements(identity, results)
    return results


def prepare(run_root: Path, *, prior_exclusions: Path | None = None) -> dict:
    run_root = run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    write_json(run_root / "hardware.json", hardware_preflight())
    code = verify_code()
    write_json(run_root / "code.json", code)
    original = json.loads(
        (
            ROOT
            / "openspec/changes/integrate-tokencake-offload-agent/evidence"
            / "provenance.json"
        ).read_text()
    )
    provenance = capture(
        SimpleNamespace(
            target=ROOT,
            model=Path(workload_parameters()["model"]),
            original_environment=Path(original["original_environment"]["path"]),
            mooncake_wheel=Path(original["mooncake_wheel"]["path"]),
            vllm_wheel=Path(original["vllm_wheel"]["path"]),
        )
    )
    write_json(run_root / "provenance.json", provenance)
    launcher = materialize(SOURCE, run_root / "launcher")
    reference_launcher = materialize(
        SOURCE, run_root / "reference-launcher", patched=False
    )
    write_json(run_root / "launcher.json", launcher)
    write_json(run_root / "reference-launcher.json", reference_launcher)
    parameters = workload_parameters()
    adapter("freeze", run_root / "launcher", parameters, run_root / "workload.json")
    adapter(
        "freeze", SOURCE, parameters, run_root / "source-workload.json", source=True
    )
    workload = json.loads((run_root / "workload.json").read_text())
    if workload != json.loads((run_root / "source-workload.json").read_text()):
        raise RuntimeError(
            "Target and source generated different frozen workload inputs"
        )
    for qps, offsets in workload["arrivals"].items():
        write_json(run_root / f"arrivals-{qps}.json", {"offsets_s": offsets})
    environments = {}
    for role in ("target", "source", "baseline"):
        python = ROOT / (
            ".venv/source/.venv/bin/python" if role == "source" else ".venv/bin/python"
        )
        output = run_root / f"environment-{role}.json"
        with run_root.joinpath(f"environment-{role}.log").open("x") as log:
            subprocess.run(
                [
                    str(python),
                    str(PACKAGE / "verify_environment.py"),
                    "--role",
                    role,
                    "--checkout",
                    code[role]["checkout"],
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                env=os.environ
                | {"PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=300,
            )
        packages = json.loads(
            subprocess.check_output(
                ["uv", "pip", "list", "--python", str(python), "--format=json"],
                text=True,
            )
        )
        runtime = json.loads(output.read_text())
        environments[role] = content_hash(
            {
                "runtime": {
                    key: value for key, value in runtime.items() if key != "command"
                },
                "packages": packages,
                "base_packages": provenance["original_environment"]["packages"],
            }
        )
        write_json(run_root / f"packages-{role}.json", packages)
    inputs = content_hash(
        {
            "parameters": parameters,
            "workload": workload,
            "model": provenance["model"],
            "dataset": provenance["dataset"],
        }
    )
    identities = []
    mooncake = {
        "configuration": json.loads(
            (MOONCAKE_CONFIGURATION / "server_mooncake.json").read_text()
        )
        | MOONCAKE_SETTINGS,
        "master_sha256": digest(ROOT / ".venv/bin/mooncake_master"),
        "configuration_commit": MOONCAKE_REVISION,
    }
    inherited = inherited_environment()
    for queues in initial_queues().values():
        for queue in queues:
            for case in queue:
                role = (
                    "source"
                    if case.mode == "old-offload-agent"
                    else "baseline"
                    if case.mode == "native"
                    else "target"
                )
                helpers = (
                    launcher
                    if case.mode in ("native", "agent", "offload-agent")
                    else reference_launcher
                )
                identity = Identity(
                    case,
                    content_hash(code[role]["runtime_tree"]),
                    environments[role],
                    content_hash(
                        {
                            "helpers": helpers["materialized_helpers"],
                            "wrapper": None
                            if case.mode == "old-offload-agent"
                            else digest(PACKAGE / "launch_client.py"),
                        }
                    ),
                    inputs,
                    content_hash(
                        {
                            "server": server_command(case, 0)[:2],
                            "inherited_environment": inherited,
                            "mooncake": mooncake if case.mode == "mooncake" else None,
                        }
                    ),
                )
                identities.append(identity.payload())
    if prior_exclusions is not None:
        ledger = carry_exclusions(
            prior_exclusions.resolve(),
            [
                Identity(
                    Case(**payload["case"]),
                    **{k: v for k, v in payload.items() if k not in ("case", "key")},
                )
                for payload in identities
            ],
        )
        write_json(run_root / "prior-exclusions.json", ledger)
    frozen = {
        "parameters": parameters,
        "code": code,
        "workload_sha256": workload["workload_sha256"],
        "arrivals": workload["arrivals"],
        "identities": identities,
        "tool_hashes": {
            path.name: digest(path)
            for path in sorted(PACKAGE.iterdir())
            if path.suffix in (".py", ".patch")
        },
        "mooncake": mooncake["configuration"],
        "mooncake_master_sha256": mooncake["master_sha256"],
        "inherited_environment": inherited,
        "input_hashes": {
            path.name: digest(path) for path in sorted(run_root.glob("*.json"))
        },
    }
    write_json(run_root / "frozen.json", frozen)
    return frozen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    print(json.dumps(prepare(parser.parse_args().run_root), indent=2, sort_keys=True))
