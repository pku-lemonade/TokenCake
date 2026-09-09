# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run and audit the fixed two-A800, full-DAG acceptance campaign."""

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
from pathlib import Path

import httpx
import regex as re

from tools.tokencake_experiments.campaign import (
    COMPONENT_QPS,
    MOONCAKE_SETTINGS,
    PACKAGE,
    ROOT,
    Case,
    client_command,
    component_queues,
    initial_queues,
    server_command,
    workload_parameters,
)
from tools.tokencake_experiments.materialize import WORKLOAD_PROFILES, digest
from tools.tokencake_experiments.preflight import (
    MOONCAKE_CONFIGURATION,
    adapter,
    inherited_environment,
    prepare,
    verify_code,
)
from tools.tokencake_experiments.report import Identity, evaluate, measurements
from tools.tokencake_experiments.runtime import (
    Activity,
    MetricsMonitor,
    Monitor,
    Process,
    free_port,
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
    if identity.key != payload["key"]:
        raise ValueError("Frozen identity hash mismatch")
    return identity


def plan(*, components: bool = False) -> dict:
    queues = {}
    selected = {"components": component_queues()} if components else initial_queues()
    for stage, stage_queues in selected.items():
        queues[stage] = []
        for queue in stage_queues:
            entries = []
            for case in queue:
                port = 8055 + case.device.index
                command, environment, cwd = server_command(case, port)
                client, client_env, client_cwd = client_command(
                    case, port, Path("CASE"), Path("LAUNCHER"), Path("ARRIVALS.json")
                )
                entries.append(
                    case.payload()
                    | {
                        "server_command": command,
                        "server_environment": environment,
                        "server_cwd": str(cwd),
                        "client_command": client,
                        "client_environment": client_env,
                        "client_cwd": str(client_cwd),
                    }
                )
            queues[stage].append(entries)
    parameters = workload_parameters()
    if components:
        parameters["qps"] = list(COMPONENT_QPS)
    return {
        "parameters": parameters,
        "queues": queues,
        "mooncake_settings": MOONCAKE_SETTINGS,
        "initial_case_count": 20 if components else 15,
        "maximum_launches": 3,
        "mooncake_launches_per_qps": 1,
        "thermal_validity_enforced": False,
        "maximum_concurrent_host_offload_servers": 1,
    }


def audit(case_dir: Path, source_result: dict, execution: dict, frozen: dict) -> dict:
    """Combine the unchanged DAG analyzer with current runtime evidence."""
    result = source_result.copy()
    reasons = list(result.get("exclusion_reasons", []))
    if execution["client_exit_code"] != 0:
        reasons.append("client_failure")
    if not execution["server_alive"]:
        reasons.append("server_failure")
    if execution.get("error"):
        reasons.append("execution_failure")
    if execution["contamination"].get("affinity_violation"):
        reasons.append("affinity_violation")
    if execution["contamination"].get("external_gpu_process_detected"):
        reasons.append("external-gpu-process")
    if execution["contamination"].get("monitor_query_failed"):
        reasons.append("contamination-monitor-failure")
    if (
        result.get("correctness", {}).get("observed_workload_hash")
        != frozen["workload_sha256"]
    ):
        reasons.append("workload_mismatch")
    app_files = list((case_dir / "app_results").glob("*.json"))
    if len(app_files) == 1:
        applications = json.loads(app_files[0].read_text())
        qps = execution["identity"]["case"]["qps"]
        if any(
            applications.get(str(index), {}).get("arrival_offset_s") != offset
            for index, offset in enumerate(frozen["arrivals"][str(float(qps))])
        ):
            reasons.append("arrival_schedule_mismatch")
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
        # The source analyzer recognizes old log telemetry. The current native
        # connector exposes the same operation outcomes through Prometheus.
        if puts > 0:
            reasons = [
                reason for reason in reasons if reason != "mooncake-no-store-operations"
            ]
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
        self.identities = [
            load_identity(payload) for payload in self.frozen["identities"]
        ]
        self.by_case = {identity.case.name: identity for identity in self.identities}
        self.activity = Activity()
        self.lock = threading.Lock()
        self.offload_lock = threading.Lock()
        self.stop = threading.Event()
        self.results = self._load_results()
        self.triggered: set[str] = set()
        for path in sorted(self.root.glob("reports/*.json")):
            self.triggered.update(
                json.loads(path.read_text()).get("triggered_gates", [])
            )

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

    def verify_frozen(self) -> None:
        for name, expected in self.frozen["input_hashes"].items():
            if digest(self.root / name) != expected:
                raise RuntimeError(f"Frozen campaign input changed: {name}")
        if inherited_environment() != self.frozen["inherited_environment"]:
            raise RuntimeError("Inherited serving environment changed")
        target = self.frozen["code"]["target"]
        code = verify_code(
            target_snapshot=Path(target["checkout"]) if target.get("snapshot") else None
        )
        for role, recorded in self.frozen["code"].items():
            if role != "target" and code[role] != recorded:
                raise RuntimeError(f"Frozen {role} code changed")
            if (
                role == "target"
                and code[role]["runtime_tree"] != recorded["runtime_tree"]
            ):
                raise RuntimeError("Target executable changed during the campaign")
        for name, expected in self.frozen["tool_hashes"].items():
            if digest(PACKAGE / name) != expected:
                raise RuntimeError(f"Frozen experiment tool changed: {name}")
        for name in ("launcher", "reference-launcher"):
            manifest = json.loads((self.root / f"{name}.json").read_text())
            for relative, expected in manifest["materialized_helpers"].items():
                if digest(self.root / name / relative) != expected:
                    raise RuntimeError(f"Materialized launcher changed: {relative}")

    def verify_inputs(self) -> None:
        provenance = json.loads((self.root / "provenance.json").read_text())
        base = provenance["original_environment"]
        packages = json.loads(
            subprocess.check_output(
                [
                    "uv",
                    "pip",
                    "list",
                    "--python",
                    str(Path(base["path"]) / "bin/python"),
                    "--format=json",
                ],
                text=True,
            )
        )
        if packages != base["packages"]:
            raise RuntimeError("Read-only base environment packages changed")
        files = [
            provenance["dataset"],
            *provenance["model"]["files"],
            provenance["mooncake_wheel"],
            provenance["vllm_wheel"],
        ]
        for role in ("target", "source", "baseline"):
            runtime = json.loads((self.root / f"environment-{role}.json").read_text())
            files.append(runtime["extension"])
            if "mooncake" in runtime:
                files.extend(runtime["mooncake"][key] for key in ("engine", "store"))
            packages = json.loads(
                subprocess.check_output(
                    [
                        "uv",
                        "pip",
                        "list",
                        "--python",
                        runtime["executable"],
                        "--format=json",
                    ],
                    text=True,
                )
            )
            if packages != json.loads(
                (self.root / f"packages-{role}.json").read_text()
            ):
                raise RuntimeError(f"Frozen {role} packages changed")
        for entry in files:
            if digest(Path(entry["path"])) != entry["sha256"]:
                raise RuntimeError(f"Frozen input changed: {entry['path']}")
        if (
            digest(ROOT / ".venv/bin/mooncake_master")
            != self.frozen["mooncake_master_sha256"]
        ):
            raise RuntimeError("Mooncake master executable changed")

    def _capture(self, port: int, name: str, destination: Path, **params: str) -> None:
        with httpx.Client(timeout=120) as client:
            response = client.get(f"http://127.0.0.1:{port}/{name}", params=params)
            response.raise_for_status()
        with destination.open("x") as stream:
            stream.write(response.text)

    def run_case(self, identity: Identity) -> dict:
        if self.stop.is_set():
            raise InterruptedError("Campaign interrupted")
        self.verify_frozen()
        case = identity.case
        with self.lock:
            launch = len(measurements(identity, self.results))
        if launch >= (1 if case.mode == "mooncake" else 3):
            raise RuntimeError(f"Launch budget exhausted: {case.name}")
        case_dir = self.root / "cases" / case.name / f"launch-{launch}"
        case_dir.mkdir(parents=True, exist_ok=False)
        port = free_port(8055 + case.device.index)
        command, environment, cwd = server_command(
            case,
            port,
            target_checkout=Path(self.frozen["code"]["target"]["checkout"]),
        )
        checkout = self.root / (
            "reference-launcher" if case.mode == "mooncake" else "launcher"
        )
        client_cmd, client_environment, client_cwd = client_command(
            case, port, case_dir, checkout, self.root / f"arrivals-{case.qps}.json"
        )
        environment["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + os.environ["PATH"]
        environment["VLLM_CACHE_ROOT"] = "/root/autodl-tmp/tokencake-vllm-cache"
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
        self.activity.set(case.device, f"{case.name}/launch-{launch}", roots)
        execution = attempt | {
            "client_exit_code": None,
            "server_alive": False,
            "error": None,
            "client_started_at": None,
            "client_finished_at": None,
            "performance": {},
        }
        total_e2e_s = 0.0
        monitor = Monitor(
            case.device, self.activity, case_dir / "gpu-process-thermal.jsonl"
        )
        print(f"START {case.name} launch={launch} gpu={case.device.uuid}", flush=True)
        try:
            if case.mode == "mooncake":
                master_port = free_port(50123)
                configuration = json.loads(
                    (MOONCAKE_CONFIGURATION / "server_mooncake.json").read_text()
                )
                configuration.update(self.frozen["mooncake"])
                configuration["master_server_address"] = f"127.0.0.1:{master_port}"
                config_path = case_dir / "mooncake_config.json"
                write_json(config_path, configuration)
                environment["MOONCAKE_CONFIG_PATH"] = str(config_path)
                master_cmd = [
                    "taskset",
                    "-c",
                    case.device.cpus,
                    str(ROOT / ".venv/bin/mooncake_master"),
                    f"--port={master_port}",
                    "--logtostderr=1",
                ]
                write_json(case_dir / "mooncake_master_command.json", master_cmd)
                master = Process(
                    master_cmd,
                    environment,
                    cwd,
                    case_dir / "mooncake_master.log",
                    cpus=case.device.cpus,
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
                cpus=case.device.cpus,
            )
            processes.append(server)
            roots.append(server.process.pid)
            self.activity.set(case.device, f"{case.name}/launch-{launch}", roots)
            with monitor:
                wait_ready(server, port, stop=self.stop)
                self._capture(port, "metrics", case_dir / "metrics-before.prom")
                initial_samples = metric_values(
                    (case_dir / "metrics-before.prom").read_text()
                )
                initial_requests = metric_sum(
                    initial_samples, "vllm:request_success_total"
                )
                if initial_requests:
                    raise RuntimeError("A fresh server already processed requests")
                if case.mode == "old-offload-agent":
                    self._capture(port, "v1/mcp/debug", case_dir / "state-before.json")
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
                    cpus=case.device.cpus,
                )
                processes.append(client)
                roots.append(client.process.pid)
                self.activity.set(case.device, f"{case.name}/launch-{launch}", roots)
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
                if case.mode == "old-offload-agent":
                    self._capture(port, "v1/mcp/debug", case_dir / "state_after.json")
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
            self.activity.clear(case.device)
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
                self.root / "reference-launcher",
                {
                    "case_dir": str(case_dir),
                    "total_e2e_s": total_e2e_s,
                    "attempt": {
                        "point": {"num_requests": 24, "qps": case.qps},
                        "mode_config": {
                            "kind": "mooncake"
                            if case.mode == "mooncake"
                            else "tokencake"
                        },
                    },
                    "contamination": monitor.summary,
                    "old_protocol": case.mode == "old-offload-agent",
                    "state_isolation": {
                        "passed": "server_pid" in execution,
                        "fresh_process": True,
                    },
                },
                case_dir / "source-analysis.json",
            )
            source_result = json.loads((case_dir / "source-analysis.json").read_text())
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
                if case.mode in ("offload", "offload-agent", "old-offload-agent"):
                    # Two 100 GiB pinned pools exceed this container's budget
                    # once CUDA, model loading, and allocator overhead are included.
                    with self.offload_lock:
                        self.run_case(identity)
                else:
                    self.run_case(identity)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_queue, queue) for queue in queues]
            for future in futures:
                future.result()

    def report(self, *, include_old: bool) -> dict:
        report = evaluate(
            self.identities,
            self.results,
            include_old=include_old,
            previously_triggered=self.triggered,
            native_improvement=self.frozen.get("native_improvement", 0.10),
            qps_values=self.frozen.get("parameters", {}).get("qps", (1.0, 0.5, 0.1)),
        )
        self.triggered.update(report["triggered_gates"])
        number = len(list(self.root.glob("reports/*.json")))
        write_json(self.root / "reports" / f"{number:04d}.json", report)
        return report

    def repeat_affected(self, *, include_old: bool) -> dict:
        while True:
            report = self.report(include_old=include_old)
            if self.stop.is_set() or not report["requested_launches"]:
                return report
            queues: list[list[Case]] = [[], []]
            for identity in self.identities:
                count = report["requested_launches"].get(identity.key, 0)
                queues[identity.case.device.index].extend([identity.case] * count)
            for queue in queues:
                queue.sort(key=lambda case: (-case.qps, case.mode))
            self.run_queues(queues, initial=False)

    def run(self) -> dict:
        self.verify_frozen()
        self.verify_inputs()
        self.run_queues(initial_queues()["primary"], initial=True)
        self.repeat_affected(include_old=False)
        self.run_queues(initial_queues()["references"], initial=True)
        return self.repeat_affected(include_old=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    planning = subparsers.add_parser("plan")
    planning.add_argument("--output", type=Path)
    planning.add_argument("--components", action="store_true")
    for command in ("prepare", "run", "run-cases", "report"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("run_root", type=Path)
        if command == "prepare":
            subparser.add_argument("--prior-exclusions", type=Path)
            subparser.add_argument("--snapshot-target", action="store_true")
            subparser.add_argument("--components", action="store_true")
            subparser.add_argument(
                "--workload-profile", choices=WORKLOAD_PROFILES, default="frozen"
            )
            subparser.add_argument("--workload-dataset", type=Path)
            subparser.add_argument(
                "--mode",
                choices=("native", "agent", "offload", "offload-agent"),
                action="append",
            )
            subparser.add_argument(
                "--qps", type=float, choices=COMPONENT_QPS, action="append"
            )
            subparser.add_argument("--gpu", type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.command == "plan":
        result = plan(components=args.components)
        if args.output:
            write_json(args.output, result)
    elif args.command == "prepare":
        if args.components and (args.mode or args.qps or args.gpu is not None):
            parser.error("--components cannot be combined with --mode, --qps or --gpu")
        if (args.qps or args.gpu is not None) and not args.mode:
            parser.error("--qps and --gpu require --mode")
        cases = (
            [
                Case(mode, qps, gpu_index=args.gpu)
                for mode in args.mode
                for qps in sorted(args.qps or [1.0, 0.5, 0.1], reverse=True)
            ]
            if args.mode
            else None
        )
        if args.components:
            cases = [case for queue in component_queues() for case in queue]
        result = prepare(
            args.run_root,
            prior_exclusions=args.prior_exclusions,
            snapshot_target=args.snapshot_target,
            cases=cases,
            workload_profile=args.workload_profile,
            workload_dataset=args.workload_dataset,
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
                    runner.verify_frozen()
                    runner.verify_inputs()
                    queues: list[list[Case]] = [[], []]
                    for identity in runner.identities:
                        queues[identity.case.device.index].append(identity.case)
                    runner.run_queues(queues, initial=True)
                    result = {"results": runner.results}
            else:
                result = runner.report(include_old=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
