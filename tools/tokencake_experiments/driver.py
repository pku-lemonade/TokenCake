# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run and audit full-DAG experiments on configurable local GPUs."""

import argparse
import concurrent.futures
import fcntl
import json
import os
import signal
import socket
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path

import httpx
import regex as re

from tools.tokencake_experiments.campaign import (
    COMPONENT_QPS,
    CURRENT_MODES,
    ROOT,
    Case,
    Device,
    Settings,
    add_settings_arguments,
    affinity_command,
    client_command,
    component_queues,
    group_cases,
    initial_queues,
    make_cases,
    server_command,
    settings_from_args,
    workload_parameters,
)
from tools.tokencake_experiments.materialize import WORKLOAD_PROFILES, digest
from tools.tokencake_experiments.preflight import (
    adapter,
    prepare,
)
from tools.tokencake_experiments.report import Identity, measurements, summarize
from tools.tokencake_experiments.runtime import (
    Activity,
    MetricsMonitor,
    Monitor,
    Process,
    free_port,
    gpu_devices,
    metric_sum,
    metric_values,
    wait_ready,
    write_json,
)


def load_identity(payload: dict) -> Identity:
    identity = Identity(
        Case(**payload["case"]),
        **{key: value for key, value in payload.items() if key not in ("case", "key")},
    )
    return identity


def plan(*, components: bool = False, settings=None) -> dict:
    settings = settings or Settings()
    queues = {}
    selected = (
        {"components": component_queues(settings)}
        if components
        else initial_queues(settings)
    )
    for stage, stage_queues in selected.items():
        queues[stage] = []
        for queue in stage_queues:
            entries = []
            for case in queue:
                port = 8055 + settings.device(case).index
                command, environment, cwd = server_command(
                    case, port, settings=settings
                )
                client, client_env, client_cwd = client_command(
                    case,
                    port,
                    Path("CASE"),
                    Path("LAUNCHER"),
                    Path("ARRIVALS.json"),
                    settings=settings,
                )
                entries.append(
                    case.payload()
                    | {
                        "device": asdict(settings.device(case)),
                        "server_command": command,
                        "server_environment": environment,
                        "server_cwd": str(cwd),
                        "client_command": client,
                        "client_environment": client_env,
                        "client_cwd": str(client_cwd),
                    }
                )
            queues[stage].append(entries)
    parameters = workload_parameters(settings)
    if components:
        parameters["qps"] = list(COMPONENT_QPS)
    return {
        "parameters": parameters,
        "queues": queues,
        "initial_case_count": sum(
            len(queue) for stage in queues.values() for queue in stage
        ),
        "thermal_validity_enforced": False,
    }


def audit(case_dir: Path, source_result: dict, execution: dict, frozen: dict) -> dict:
    """Combine repository-local DAG validation with runtime evidence."""
    result = source_result.copy()
    reasons = list(result.get("exclusion_reasons", []))
    if execution["client_exit_code"] != 0:
        reasons.append("client_failure")
    if not execution["server_alive"]:
        reasons.append("server_failure")
    if execution.get("error"):
        reasons.append("execution_failure")
    client_log = (
        (case_dir / "client.log").read_text(errors="replace")
        if (case_dir / "client.log").exists()
        else ""
    )
    attempts = [
        json.loads(line.partition("[TokenCakeAttempt] ")[2])
        for line in client_log.splitlines()
        if line.startswith("[TokenCakeAttempt] ")
    ]
    if "[SMOKE]" in client_log:
        reasons.append("truncated_workload")
    samples = (
        metric_values((case_dir / "metrics.prom").read_text())
        if (case_dir / "metrics.prom").exists()
        else []
    )
    mode = execution["identity"]["case"]["mode"]
    if mode == "mooncake":
        puts = metric_sum(
            samples,
            "vllm:mooncake_store_operation_total",
            operation="save_put",
            status="ok",
        )
        errors = metric_sum(
            samples, "vllm:mooncake_store_operation_total", status="error"
        )
        failed_keys = metric_sum(
            samples, "vllm:mooncake_store_operation_failed_keys_total"
        )
        if puts > 0:
            reasons = [
                reason for reason in reasons if reason != "mooncake-no-store-operations"
            ]
        if puts <= 0:
            reasons.append("mooncake-no-store-operations")
        if errors or failed_keys:
            reasons.append("mooncake_native_connector_error")
        result["native_mooncake_telemetry"] = {
            "successful_put_calls": puts,
            "error_calls": errors,
            "failed_keys": failed_keys,
        }
    failures = re.findall(r"retrying\.\.\. \((\d+)/3\)", client_log)
    result["attempt_diagnostics"] = {
        "requests": attempts,
        "retries": (
            sum(attempt["attempt"] > 0 for attempt in attempts)
            if attempts
            else sum(int(index) <= 3 for index in failures)
        ),
        "prompt_halvings": len(failures),
        "legacy_retry_messages": client_log.count("retrying..."),
    }
    result["native_metrics"] = [
        sample
        for sample in samples
        if any(
            area in sample["name"]
            for area in (
                "tokencake",
                "kv_offload",
                "mooncake",
                "prefix_cache",
                "preemption",
            )
        )
    ]
    performance = result.get("performance", {}) | execution["performance"]
    result.update(execution)
    result["performance"] = performance
    result["exclusion_reasons"] = sorted(set(reasons))
    result["qualifying"] = not reasons
    result["status"] = "qualifying" if result["qualifying"] else "excluded"
    result["result_path"] = str(case_dir / "result.json")
    result["artifacts"] = {
        str(path.relative_to(case_dir)): digest(path)
        for path in sorted(case_dir.rglob("*"))
        if path.is_file()
    }
    return result


class Runner:
    def __init__(self, run_root: Path) -> None:
        self.root = run_root.resolve()
        self.frozen = json.loads((self.root / "frozen.json").read_text())
        self.settings = Settings(**self.frozen.get("settings", {}))
        self.devices = {
            row["index"]: Device(**row) for row in self.frozen.get("devices", [])
        }
        self.identities = [
            load_identity(payload) for payload in self.frozen["identities"]
        ]
        self.by_case = {identity.case.name: identity for identity in self.identities}
        self.activity = Activity()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.results = self._load_results()

    def _load_results(self) -> list[dict]:
        previous = self.root / "prior-exclusions.json"
        results = json.loads(previous.read_text()) if previous.exists() else []
        for attempt_file in sorted(self.root.glob("cases/**/attempt.json")):
            path = attempt_file.with_name("result.json")
            if not path.exists():
                attempt = json.loads(attempt_file.read_text())
                result = attempt | {
                    "qualifying": False,
                    "status": "excluded",
                    "exclusion_reasons": ["interrupted_attempt"],
                    "result_path": str(path),
                }
                result["artifacts"] = {
                    str(item.relative_to(path.parent)): digest(item)
                    for item in sorted(path.parent.rglob("*"))
                    if item.is_file()
                }
                write_json(path, result)
                results.append(result)
            else:
                results.append(json.loads(path.read_text()))
        return results

    def device(self, case):
        selected = self.settings.device(case)
        observed = {int(row["index"]): row for row in gpu_devices()}
        return Device(
            selected.index,
            observed.get(selected.index, {}).get("uuid", ""),
            cpus=selected.cpus,
        )

    def _capture(self, port: int, name: str, destination: Path, **params: str) -> None:
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(f"http://127.0.0.1:{port}/{name}", params=params)
                response.raise_for_status()
            body = response.text
        except httpx.HTTPError as exc:
            destination.with_suffix(destination.suffix + ".error.txt").write_text(
                str(exc)
            )
            body = '{"vllm_config": {}}' if name == "server_info" else ""
        with destination.open("x") as stream:
            stream.write(body)

    def run_case(self, identity: Identity) -> dict:
        if self.stop.is_set():
            raise InterruptedError("Campaign interrupted")
        case = identity.case
        device = self.device(case)
        with self.lock:
            launch = len(measurements(identity, self.results))
        case_dir = self.root / "cases" / case.name / f"launch-{launch}"
        case_dir.mkdir(parents=True, exist_ok=False)
        port = free_port(8055 + device.index)
        command, environment, cwd = server_command(
            case,
            port,
            target_checkout=(
                self.root / "runtime"
                if self.frozen["code"]["target"].get("snapshot")
                else ROOT
            ),
            settings=self.settings,
            device=device,
        )
        checkout = self.root / "launcher"
        client_cmd, client_environment, client_cwd = client_command(
            case,
            port,
            case_dir,
            checkout,
            self.root / f"arrivals-{float(case.qps)}.json",
            workload=self.frozen["parameters"],
            settings=self.settings,
            device=device,
        )
        environment["PATH"] = (
            str(Path(self.settings.python).parent) + os.pathsep + os.environ["PATH"]
        )
        environment["VLLM_CACHE_ROOT"] = str(self.root / "cache/vllm")
        client_environment["PATH"] = environment["PATH"]
        attempt = {
            "identity": identity.payload(),
            "launch": launch,
            "started_at": time.time(),
        }
        write_json(case_dir / "attempt.json", attempt)
        processes: list[Process] = []
        metrics_monitor: MetricsMonitor | None = None
        server: Process | None = None
        roots: list[int] = []
        self.activity.set(device, f"{case.name}/launch-{launch}", roots)
        execution = attempt | {
            "client_exit_code": None,
            "server_alive": False,
            "error": None,
            "client_started_at": None,
            "client_finished_at": None,
            "performance": {},
        }
        total_e2e_s = 0.0
        monitor = Monitor(device, self.activity, case_dir / "gpu-process-thermal.jsonl")
        print(f"START {case.name} launch={launch} gpu={device.uuid}", flush=True)
        try:
            if case.mode == "mooncake" and self.frozen.get("mooncake_master"):
                master_port = free_port(50123)
                configuration = dict(self.frozen["mooncake"])
                configuration["master_server_address"] = f"127.0.0.1:{master_port}"
                config_path = case_dir / "mooncake_config.json"
                write_json(config_path, configuration)
                environment["MOONCAKE_CONFIG_PATH"] = str(config_path)
                master_cmd = affinity_command(
                    [
                        self.frozen["mooncake_master"]["path"],
                        f"--port={master_port}",
                        "--logtostderr=1",
                    ],
                    device.cpus,
                )
                write_json(case_dir / "mooncake_master_command.json", master_cmd)
                master = Process(
                    master_cmd,
                    environment,
                    cwd,
                    case_dir / "mooncake_master.log",
                    cpus=device.cpus,
                )
                processes.append(master)
                roots.append(master.process.pid)
                deadline = time.monotonic() + 30
                while True:
                    if self.stop.is_set():
                        raise InterruptedError("Campaign interrupted")
                    if master.process.poll() is not None:
                        raise RuntimeError("Mooncake master exited before readiness")
                    try:
                        with socket.create_connection(
                            ("127.0.0.1", master_port), timeout=1
                        ):
                            break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "Mooncake master readiness timed out"
                            ) from None
                        time.sleep(0.2)
            write_json(
                case_dir / "commands.json",
                {
                    "server": command,
                    "server_environment": environment,
                    "server_cwd": str(cwd),
                    "client": client_cmd,
                    "client_environment": client_environment,
                    "client_cwd": str(client_cwd),
                    "provenance": str(self.root / "provenance.json"),
                    "provenance_sha256": digest(self.root / "provenance.json"),
                },
            )
            server = Process(
                command,
                environment,
                cwd,
                case_dir / "server.log",
                cpus=device.cpus,
            )
            processes.append(server)
            roots.append(server.process.pid)
            self.activity.set(device, f"{case.name}/launch-{launch}", roots)
            with monitor:
                wait_ready(server, port, stop=self.stop)
                self._capture(port, "metrics", case_dir / "metrics-before.prom")
                self._capture(
                    port,
                    "server_info",
                    case_dir / "resolved-server.json",
                    config_format="json",
                )
                execution["server_pid"] = server.process.pid
                metrics_monitor = MetricsMonitor(
                    port, case_dir / "metrics-timeseries.jsonl"
                )
                metrics_monitor.__enter__()
                execution["client_started_at"] = time.time()
                start = time.monotonic()
                execution["client_start_monotonic"] = start
                client = Process(
                    client_cmd,
                    client_environment,
                    client_cwd,
                    case_dir / "client.log",
                    cpus=device.cpus,
                )
                processes.append(client)
                roots.append(client.process.pid)
                self.activity.set(device, f"{case.name}/launch-{launch}", roots)
                while True:
                    try:
                        execution["client_exit_code"] = client.process.wait(timeout=2)
                        break
                    except subprocess.TimeoutExpired:
                        if self.stop.is_set():
                            raise InterruptedError("Campaign interrupted") from None
                        if server.process.poll() is not None:
                            raise RuntimeError(
                                "Server exited during workload execution"
                            ) from None
                        if time.monotonic() - start >= 7200:
                            raise TimeoutError(
                                "Complete workload exceeded the case timeout"
                            ) from None
                total_e2e_s = time.monotonic() - start
                execution["client_end_monotonic"] = start + total_e2e_s
                execution["client_finished_at"] = time.time()
                execution["server_alive"] = server.process.poll() is None
                self._capture(port, "health", case_dir / "health-after.txt")
                self._capture(port, "metrics", case_dir / "metrics.prom")
        except Exception as exc:
            execution["error"] = f"{type(exc).__name__}: {exc}"
            if server is not None:
                execution["server_alive"] = server.process.poll() is None
        finally:
            if metrics_monitor is not None:
                try:
                    metrics_monitor.__exit__()
                except Exception as exc:
                    execution["error"] = f"Metrics monitor cleanup failed: {exc}"
                execution["metrics_monitor"] = metrics_monitor.summary
            for process in reversed(processes):
                try:
                    process.close()
                except Exception as exc:
                    execution["error"] = f"Process cleanup failed: {exc}"
            self.activity.clear(device)
        if (
            execution["client_started_at"] is not None
            and execution["client_finished_at"] is None
        ):
            execution["client_finished_at"] = time.time()
            execution["client_end_monotonic"] = time.monotonic()
            total_e2e_s = (
                execution["client_end_monotonic"] - execution["client_start_monotonic"]
            )
        execution["performance"] = {"total_e2e_s": total_e2e_s}
        execution["contamination"] = monitor.summary
        write_json(case_dir / "execution.json", execution)
        source_result = {"exclusion_reasons": ["missing_complete_dag_evidence"]}
        try:
            adapter(
                "analyze",
                self.root / "launcher",
                {
                    "case_dir": str(case_dir),
                    "total_e2e_s": total_e2e_s,
                    "attempt": {
                        "point": {
                            "num_requests": self.frozen["parameters"]["num_requests"],
                            "qps": case.qps,
                        },
                        "mode_config": {
                            "kind": "mooncake"
                            if case.mode == "mooncake"
                            else "tokencake"
                        },
                    },
                    "contamination": monitor.summary,
                    "state_isolation": {
                        "passed": "server_pid" in execution,
                        "fresh_process": True,
                    },
                },
                case_dir / "analysis.json",
                python=self.settings.python,
            )
            source_result = json.loads((case_dir / "analysis.json").read_text())
        except Exception as exc:
            execution["analysis_error"] = f"{type(exc).__name__}: {exc}"
        result = audit(case_dir, source_result, execution, self.frozen)
        write_json(case_dir / "result.json", result)
        with self.lock:
            self.results.append(result)
        print(
            f"END {case.name} launch={launch} {result['status']} "
            f"total_e2e_s={total_e2e_s:.3f} reasons={result['exclusion_reasons']}",
            flush=True,
        )
        return result

    def run_queues(self, queues: list[list[Case]], *, initial: bool) -> None:
        def run_queue(queue: list[Case]) -> None:
            for case in queue:
                if self.stop.is_set():
                    return
                identity = self.by_case[case.name]
                if initial and any(
                    "budget_identity" not in row
                    for row in measurements(identity, self.results)
                ):
                    continue
                self.run_case(identity)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(queues))
        ) as pool:
            futures = [pool.submit(run_queue, queue) for queue in queues]
            for future in futures:
                future.result()

    def report(self) -> dict:
        report = summarize(self.identities, self.results)
        number = len(list(self.root.glob("reports/*.json")))
        write_json(self.root / "reports" / f"{number:04d}.json", report)
        return report

    def run(self) -> dict:
        self.run_queues(
            group_cases([identity.case for identity in self.identities], self.settings),
            initial=True,
        )
        return self.report()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    planning = subparsers.add_parser("plan")
    planning.add_argument("--output", type=Path)
    planning.add_argument("--components", action="store_true")
    add_settings_arguments(planning)
    for command in ("prepare", "run", "run-cases", "report"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("run_root", type=Path)
        if command == "prepare":
            add_settings_arguments(subparser)
            subparser.add_argument("--mooncake-config", type=Path)
            subparser.add_argument("--mooncake-master", type=Path)
            subparser.add_argument("--prior-exclusions", type=Path)
            subparser.add_argument("--snapshot-target", action="store_true")
            subparser.add_argument("--components", action="store_true")
            subparser.add_argument(
                "--workload-profile",
                choices=WORKLOAD_PROFILES,
                default="conversation-tools",
            )
            subparser.add_argument("--workload-dataset", type=Path)
            subparser.add_argument(
                "--mode",
                choices=CURRENT_MODES,
                action="append",
            )
            subparser.add_argument("--qps", type=float, action="append")
    args = parser.parse_args()
    if args.command == "plan":
        result = plan(components=args.components, settings=settings_from_args(args))
        if args.output:
            write_json(args.output, result)
    elif args.command == "prepare":
        settings = settings_from_args(args)
        if args.components and (args.mode or args.qps):
            parser.error("--components cannot be combined with --mode or --qps")
        if args.qps and not args.mode:
            parser.error("--qps requires --mode")
        cases = (
            make_cases(
                settings, args.mode, sorted(args.qps or [1.0, 0.5, 0.1], reverse=True)
            )
            if args.mode
            else None
        )
        if args.components:
            cases = [case for queue in component_queues(settings) for case in queue]
        result = prepare(
            args.run_root,
            prior_exclusions=args.prior_exclusions,
            snapshot_target=args.snapshot_target,
            cases=cases,
            workload_profile=args.workload_profile,
            workload_dataset=args.workload_dataset,
            settings=settings,
            mooncake_config=args.mooncake_config,
            mooncake_master=args.mooncake_master,
        )
    else:
        # Hold the ledger lock for reports too, so a live attempt cannot be
        # mistaken for an interrupted one by a concurrent invocation.
        with (args.run_root / "campaign.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            runner = Runner(args.run_root)
            if args.command in ("run", "run-cases"):

                def interrupt(signum, frame):
                    runner.stop.set()

                for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                    signal.signal(signum, interrupt)
                if args.command == "run":
                    result = runner.run()
                else:
                    runner.run_queues(
                        group_cases(
                            [identity.case for identity in runner.identities],
                            runner.settings,
                        ),
                        initial=True,
                    )
                    result = {"results": runner.results}
            else:
                result = runner.report()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
