# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate completed JSON DAGs and summarize their measured performance."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean


def content_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def validate_dag_completion(applications, server_log, expected_count, *, dataset=None):
    reasons = []
    statuses = Counter()
    tokens = Counter(prompt=0, generated=0, processed=0)
    contracts = []
    expected_apps = {app.id: app for app in dataset.applications} if dataset else {}
    if len(applications) != expected_count or (
        dataset and set(applications) != set(expected_apps)
    ):
        reasons.append("application_count_mismatch")
    nodes = {node.name: node for node in dataset.nodes} if dataset else {}
    order = list(expected_apps) if dataset else sorted(applications)
    for key in dict.fromkeys([*order, *applications]):
        if key not in applications:
            continue
        app = applications[key]
        contract = app.get("frozen_workload_contract")
        contracts.append(contract)
        if (
            not app.get("app_finished")
            or app.get("application_internal_finished") is False
        ):
            reasons.append("incomplete_application")
        latency = app.get("app_latency")
        if (
            not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency < 0
        ):
            reasons.append("invalid_application_latency")
        records = app.get("request_info", {})
        expected = set(nodes) if dataset else set(app.get("expected_node_names", []))
        if not expected or set(records) != expected:
            reasons.append("dag_node_mismatch")
        if dataset and (
            key not in expected_apps or contract != dataset.contract(expected_apps[key])
        ):
            reasons.append("workload_contract_mismatch")
        for name, record in records.items():
            kind = record.get("execution_kind")
            finish = record.get("finish_reason")
            node = nodes.get(name)
            if node and kind != ("llm" if node.kind == "llm" else "local"):
                reasons.append("node_execution_kind_mismatch")
            if kind == "local" and finish == "local":
                statuses["FINISHED_LOCAL"] += 1
            elif kind == "llm" and finish == "length":
                statuses["FINISHED_LENGTH_CAPPED"] += 1
            else:
                statuses[f"INVALID_{finish}"] += 1
                reasons.append("invalid_terminal_status")
            for source, target in (
                ("prompt_tokens", "prompt"),
                ("generated_tokens", "generated"),
                ("processed_tokens", "processed"),
            ):
                value = record.get(source, 0)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    reasons.append("invalid_token_count")
                else:
                    tokens[target] += value
            if (
                node
                and kind == "llm"
                and record.get("generated_tokens") != node.max_tokens
            ):
                reasons.append("generation_budget_mismatch")
    if any(
        marker in server_log
        for marker in (
            "Traceback (most recent call last)",
            "CUDA error",
            "CUDA out of memory",
        )
    ):
        reasons.append("server_error")
    return {
        "passed": not reasons,
        "exclusion_reasons": sorted(set(reasons)),
        "application_count": len(applications),
        "observed_workload_hash": content_hash(contracts),
        "token_counts": dict(tokens),
        "terminal_status_counts": dict(statuses),
        "finished_preempted_count": 0,
        "output_text_compared": False,
    }


def summarize_attempt(
    case_dir, attempt, total_e2e_s, contamination, state_isolation, *, dataset=None
):
    case_dir = Path(case_dir)
    paths = list((case_dir / "app_results").glob("*.json"))
    if len(paths) != 1:
        return {
            "correctness": {"passed": False},
            "exclusion_reasons": ["missing_complete_dag_evidence"],
        }
    apps = json.loads(paths[0].read_text())
    server_log = (case_dir / "server.log").read_text(errors="replace")
    correctness = validate_dag_completion(
        apps, server_log, attempt["point"]["num_requests"], dataset=dataset
    )
    reasons = list(correctness["exclusion_reasons"])
    if not state_isolation.get("passed"):
        reasons.append("state_isolation_failed")
    log = (case_dir / "client.log").read_text(errors="replace")
    if "retrying..." in log:
        reasons.append("request_retries")
    if not math.isfinite(total_e2e_s) or total_e2e_s <= 0:
        reasons.append("invalid_total_e2e")
    latencies = [
        app["app_latency"]
        for app in apps.values()
        if isinstance(app.get("app_latency"), (int, float))
        and math.isfinite(app["app_latency"])
    ]
    return {
        "correctness": correctness,
        "exclusion_reasons": sorted(set(reasons)),
        "performance": {
            "total_e2e_s": total_e2e_s,
            "completed_applications": sum(
                bool(app.get("app_finished")) for app in apps.values()
            ),
            "max_app_latency_s": max(latencies) if latencies else None,
            "average_app_latency_s": mean(latencies) if latencies else None,
            "p50_app_latency_s": percentile(latencies, 0.5),
            "p90_app_latency_s": percentile(latencies, 0.9),
            "p95_app_latency_s": percentile(latencies, 0.95),
            "p99_app_latency_s": percentile(latencies, 0.99),
            "throughput_rps": len(apps) / total_e2e_s if total_e2e_s > 0 else None,
        },
    }
