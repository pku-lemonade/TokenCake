# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process failure, deadline, grading isolation, and complete-denominator checks."""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.tokencake_experiments.agent_bench import experiment, swe_local, worker
from tools.tokencake_experiments.agent_bench import parallel as paired
from tools.tokencake_experiments.agent_bench.budget import Budget
from tools.tokencake_experiments.agent_bench.inputs import digest
from tools.tokencake_experiments.agent_bench.report import (
    build_report,
    distribution,
    task_observations,
)
from tools.tokencake_experiments.agent_bench.report_metrics import (
    completion_progress,
    request_length_diagnostics,
    request_records,
    server_diagnostics,
)
from tools.tokencake_experiments.agent_bench.score import score_one
from tools.tokencake_experiments.agent_bench.transport import write_json


def test_controller_launch_failure_records_active_and_unstarted_tasks(
    tmp_path, monkeypatch
):
    original = subprocess.Popen
    processes: list[subprocess.Popen] = []

    def launch(command, **kwargs):
        if processes:
            raise OSError("fixture could not launch second worker")
        process = original(
            [sys.executable, "-c", "import time; time.sleep(30)"], **kwargs
        )
        processes.append(process)
        return process

    monkeypatch.setattr(experiment.subprocess, "Popen", launch)
    summary = experiment.run_tasks(
        tmp_path,
        {"pilot": {"workers": 2, "arrival": "fixture"}},
        "bfcl",
        [{"id": str(index)} for index in range(3)],
        "base",
        tmp_path / "run",
        1,
        None,
        5,
    )
    assert summary["planned"] == summary["recorded"] == 3
    assert summary["statuses"] == {
        "controller_cancelled": 1,
        "worker_launch_error": 1,
        "not_started_controller_error": 1,
    }
    assert processes[0].poll() is not None
    assert json.loads((tmp_path / "run/1/terminal.json").read_text())["e2e_s"] >= 0


def test_twenty_tasks_complete_without_exceeding_sixteen_workers(tmp_path, monkeypatch):
    original = subprocess.Popen
    program = """
import json, sys, time
from pathlib import Path
spec = json.loads(Path(sys.argv[1]).read_text())
started = time.time()
time.sleep(0.1)
ended = time.time()
result = {'task_id': spec['context']['task_id'], 'status': 'completed',
          'fixture_start': started, 'fixture_end': ended,
          'e2e_s': ended - spec['context']['arrived_at']}
Path(sys.argv[2], 'result.json').write_text(json.dumps(result))
"""

    def launch(command, **kwargs):
        return original(
            [
                sys.executable,
                "-c",
                program,
                command[command.index("--spec") + 1],
                command[command.index("--output") + 1],
            ],
            **kwargs,
        )

    monkeypatch.setattr(experiment.subprocess, "Popen", launch)
    summary = experiment.run_tasks(
        tmp_path,
        {"pilot": {"workers": 16, "arrival": "fixture"}},
        "bfcl",
        [{"id": str(index)} for index in range(20)],
        "base",
        tmp_path / "run",
        1,
        None,
        30,
    )
    assert summary["statuses"] == {"completed": 20}
    events = []
    for path in (tmp_path / "run").glob("*/terminal.json"):
        row = json.loads(path.read_text())
        events.extend([(row["fixture_start"], 1), (row["fixture_end"], -1)])
    running = peak = 0
    for _, change in sorted(events):
        running += change
        peak = max(peak, running)
    assert running == 0
    assert 1 < peak <= 16


def test_supervisor_deadline_kills_worker_and_preserves_remaining_tasks(
    tmp_path, monkeypatch
):
    original = subprocess.Popen
    processes: list[subprocess.Popen] = []

    def launch(command, **kwargs):
        process = original(
            [sys.executable, "-c", "import time; time.sleep(30)"], **kwargs
        )
        processes.append(process)
        return process

    monkeypatch.setattr(experiment.subprocess, "Popen", launch)
    monkeypatch.setattr(experiment, "CLEANUP_GRACE_S", 0.01)
    summary = experiment.run_tasks(
        tmp_path,
        {"pilot": {"workers": 1, "arrival": "fixture"}},
        "bfcl",
        [{"id": "first"}, {"id": "second"}],
        "base",
        tmp_path / "run",
        1,
        None,
        0.05,
    )
    assert summary["statuses"] == {
        "supervisor_timeout": 1,
        "not_started_budget_exhausted": 1,
    }
    assert processes[0].poll() is not None


def test_snapshot_restores_ignored_builds_dependencies_and_caches(tmp_path):
    directory = tmp_path / "task"
    for name in ("repo", ".venv", "cache", "tmp"):
        (directory / name).mkdir(parents=True)
    (directory / "repo/ignored.so").write_bytes(b"original extension")
    (directory / ".venv/dependency.py").write_text("original dependency")
    swe_local.capture_prepared_state(directory)
    (directory / "repo/ignored.so").write_bytes(b"patched extension")
    (directory / ".venv/dependency.py").write_text("mutated dependency")
    (directory / "cache/result").write_text("old test result")
    swe_local.restore_prepared_state(directory)
    assert (directory / "repo/ignored.so").read_bytes() == b"original extension"
    assert (directory / ".venv/dependency.py").read_text() == "original dependency"
    assert not list((directory / "cache").iterdir())
    with (directory / "prepared-state.tar").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="snapshot changed"):
        swe_local.restore_prepared_state(directory)
    assert (directory / "repo/ignored.so").exists()


def test_image_system_setup_is_recorded_without_becoming_host_commands(tmp_path):
    installation = (
        "apt-get -y update && apt-get -y upgrade && apt-get install -y ffmpeg\n"
        "echo 'en_US UTF-8' > /etc/locale.gen\n"
        "locale-gen en_US.UTF-8\n"
        "python -m pip install --no-build-isolation -e ."
    )
    (tmp_path / "Dockerfile").write_text(
        "cat <<'EOF_environment' > /root/environment.yml\n"
        "dependencies:\n  - python=3.11.11=example\n"
        "  - pip=24.3.1=pyh_example\nEOF_environment\n"
        'echo "Current environment: $CONDA_DEFAULT_ENV"\ncd /testbed\n'
        + installation
        + "\n# Configure git\n"
    )
    details = swe_local.recipe(tmp_path)
    assert details["original_installation"] == installation
    assert details["system_setup_commands"] == installation.splitlines()[:3]
    assert details["locales"] == ["en_US.UTF-8"]
    command = swe_local.replace_install(
        details["local_installation"], tmp_path / ".venv/bin/python"
    )
    assert "apt-get" not in command
    assert "/etc/locale.gen" not in command
    assert "locale-gen" not in command
    assert command.endswith(" --no-build-isolation -e .")
    assert str(tmp_path / ".venv/bin/python") in command


def test_task_locale_is_usable_after_restoring_the_environment(tmp_path):
    if (
        not shutil.which("localedef")
        or not Path("/usr/share/i18n/locales/en_US").is_file()
    ):
        pytest.skip("Requires glibc locale generation inputs")
    directory = tmp_path / "task"
    for name in ("repo", ".venv/bin", "cache", "tmp", "logs"):
        (directory / name).mkdir(parents=True)
    python = directory / ".venv/bin/python"
    python.symlink_to(sys.executable)
    swe_local.prepare_locales(tmp_path, directory, {"locales": ["en_US.UTF-8"]})
    probe = [
        str(python),
        "-c",
        "import locale; print(locale.setlocale(locale.LC_ALL, 'en_US.UTF-8'))",
    ]

    def observed():
        return subprocess.check_output(
            probe,
            env=os.environ | swe_local.task_environment(tmp_path, directory),
            text=True,
        ).strip()

    assert observed() == "en_US.UTF-8"
    swe_local.capture_prepared_state(directory)
    generated = directory / ".venv/lib/locale/en_US.utf8/LC_CTYPE"
    original = generated.read_bytes()
    generated.write_bytes(b"modified by a previous task")
    swe_local.restore_prepared_state(directory)
    assert generated.read_bytes() == original
    assert observed() == "en_US.UTF-8"


def test_evaluation_locale_setup_preserves_patch_and_test_exit_code(tmp_path):
    platform_root = os.environ.get("TC_AGENT_BENCH_ROOT")
    if not platform_root:
        pytest.skip("Set TC_AGENT_BENCH_ROOT to the prepared official grader")
    assert platform_root is not None
    grading_python = str(Path(platform_root) / "environments/grading/.venv/bin/python")
    if (
        not shutil.which("localedef")
        or not Path("/usr/share/i18n/locales/en_US").is_file()
    ):
        pytest.skip("Requires glibc locale generation inputs")
    source, directory = tmp_path / "source", tmp_path / "task"
    source.mkdir()
    for name in ("repo", ".venv/bin", "cache", "tmp", "logs"):
        (directory / name).mkdir(parents=True)
    (directory / ".venv/bin/python").symlink_to(sys.executable)
    setup = swe_local.EVALUATION_LOCALE_SETUP
    original = (
        "cd /testbed\n"
        + setup
        + "\nexport LANG=en_US.UTF-8\nexport LC_ALL=en_US.UTF-8\n"
        "cat <<'PATCH_TEXT' > preserved.txt\n"
        + setup
        + "\nPATCH_TEXT\n: '>>>>> Start Test Output'\n"
        "python -c 'import locale; "
        'assert locale.setlocale(locale.LC_ALL, "") == "en_US.UTF-8"; '
        'print("output \u2026"); raise SystemExit(3)\'\n'
        ": '>>>>> End Test Output'\n"
    )
    (source / "eval.sh").write_text(original)
    (source / "Dockerfile").write_text(
        "cat <<'EOF_environment' > /root/environment.yml\n"
        "dependencies:\n  - python=3.11.11=example\n"
        "  - pip=24.3.1=pyh_example\nEOF_environment\n"
        'echo "Current environment: $CONDA_DEFAULT_ENV"\ncd /testbed\n'
        "python -m pip install -e .\n# Configure git\n"
    )
    details = swe_local.recipe(source)
    assert details["locales"] == ["en_US.UTF-8"]
    assert details["evaluation_locale_setup_commands"] == [setup]
    assert details["evaluation_script_sha256"] == digest(source / "eval.sh")
    write_json(directory / "recipe.json", details)

    def translate(script):
        return subprocess.run(
            [
                grading_python,
                "-c",
                "import sys; from pathlib import Path; "
                "from tools.tokencake_experiments.agent_bench.swe_local "
                "import evaluation_script; "
                "print(evaluation_script(sys.stdin.read(), Path(sys.argv[1])))",
                str(directory),
            ],
            input=script,
            capture_output=True,
            text=True,
            timeout=30,
        )

    missing = translate(original)
    assert missing.returncode != 0
    assert "locale was not prepared" in missing.stderr
    swe_local.prepare_locales(tmp_path, directory, details)
    swe_local.capture_prepared_state(directory)
    swe_local.restore_prepared_state(directory)
    translation = translate(original)
    assert translation.returncode == 0, translation.stderr
    translated = translation.stdout
    script = directory / "eval.sh"
    script.write_text(translated)
    assert translated.count(setup) == 1  # Only the literal heredoc content remains.
    result = subprocess.run(
        ["bash", str(script)],
        env=os.environ | swe_local.task_environment(tmp_path, directory),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    output = result.stdout.decode("utf-8")
    assert "output \u2026" in output
    assert ">>>>> Test Exit Code: 3" in output
    assert (directory / "repo/preserved.txt").read_text() == setup + "\n"
    unsupported = translate(original.replace(setup, "locale-gen fr_FR.UTF-8", 1))
    assert unsupported.returncode != 0
    assert "explicit adaptation" in unsupported.stderr


def test_conda_does_not_import_requests_from_the_task_checkout(tmp_path):
    conda = Path("/root/miniconda3/bin/conda")
    if not conda.is_file():
        pytest.skip("Requires the benchmark server's Miniconda installation")
    directory = tmp_path / "task"
    repo = directory / "repo"
    repo.mkdir(parents=True)
    (repo / "requests.py").write_text(
        "raise RuntimeError('The task source must not run in the manager')\n"
    )
    environment = swe_local.task_environment(tmp_path, directory)
    process = subprocess.run(
        [str(conda), "info", "--json"],
        cwd=directory,
        env=os.environ | swe_local.conda_process_environment(environment),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    assert "conda_version" in json.loads(process.stdout)
    assert environment["PYTHONPATH"] == str(repo)


def test_cached_repository_checks_out_frozen_content_and_rejects_changed_ref(tmp_path):
    def git(*args, cwd=None):
        return subprocess.check_output(
            ["git", *map(str, args)], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()

    source = tmp_path / "source"
    git("init", source)
    (source / "content.txt").write_text("frozen original")
    git("add", "content.txt", cwd=source)
    git(
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-m",
        "original",
        cwd=source,
    )
    original = git("rev-parse", "HEAD", cwd=source)
    (source / "content.txt").write_text("later source state")
    git(
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-am",
        "later",
        cwd=source,
    )
    later = git("rev-parse", "HEAD", cwd=source)
    task = {"instance_id": "fixture", "repo": "fixture/source", "base_commit": original}
    cache = tmp_path / "cache/swe-repositories/fixture"
    cache.mkdir(parents=True)
    git("clone", "--bare", "--no-hardlinks", source, cache / "repo.git")
    git(
        "--git-dir",
        cache / "repo.git",
        "update-ref",
        "refs/heads/frozen-base",
        original,
    )
    write_json(cache / "identity.json", task)

    checkout = tmp_path / "checkout"
    git("init", checkout)
    git(
        "fetch",
        "--depth",
        "1",
        swe_local.repository_source(tmp_path, task),
        original,
        cwd=checkout,
    )
    git("checkout", "--detach", "FETCH_HEAD", cwd=checkout)
    assert (checkout / "content.txt").read_text() == "frozen original"
    assert git("rev-parse", "HEAD", cwd=checkout) == original

    git("--git-dir", cache / "repo.git", "update-ref", "refs/heads/frozen-base", later)
    with pytest.raises(ValueError, match="cache revision changed"):
        swe_local.repository_source(tmp_path, task)


def test_application_result_survives_cleanup_failure(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: object()
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.tokencake_experiments.agent_bench.bfcl",
        SimpleNamespace(run=lambda *args: {"result": []}),
    )
    monkeypatch.setattr(
        worker.ctypes, "CDLL", lambda *args: SimpleNamespace(prctl=lambda *args: 0)
    )

    def fail_cleanup():
        raise RuntimeError("fixture cleanup failed")

    monkeypatch.setattr(worker, "cleanup_children", fail_cleanup)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        worker.run(
            {
                "benchmark": "bfcl",
                "base_url": "http://localhost:1/v1",
                "task": {},
                "context": {
                    "task_id": "fixture",
                    "agent_type": "bfcl",
                    "mode": "base",
                    "arrived_at": time.time(),
                    "arrival_offset_s": 0,
                    "deadline_at": time.time() + 60,
                },
            },
            tmp_path,
        )
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "completed"
    assert (
        json.loads((tmp_path / "cleanup.json").read_text())["error"]
        == "fixture cleanup failed"
    )


def test_grading_timeout_has_a_durable_failure_record(tmp_path, monkeypatch):
    def fail(*args):
        raise subprocess.TimeoutExpired("official-test-command", 1800)

    monkeypatch.setattr(swe_local, "_grade", fail)
    report = swe_local.grade(
        tmp_path,
        {"instance_id": "fixture"},
        tmp_path,
        {"model_patch": "patch"},
        tmp_path / "grade",
    )
    assert report["status"] == "grading_timeout"
    assert not report["resolved"]
    assert json.loads((tmp_path / "grade/report.json").read_text()) == report


def test_empty_submission_does_not_invoke_tests_or_count_as_resolved(tmp_path):
    report = score_one(
        {
            "benchmark": "swe",
            "task": {},
            "terminal": {"status": "Submitted", "prediction": {"submission": " \n"}},
        },
        tmp_path,
    )
    assert report["status"] == "empty_patch"
    assert not report["resolved"]
    assert not list(tmp_path.iterdir())


def test_budget_is_shared_and_unclean_measurements_require_reconciliation(tmp_path):
    path = tmp_path / "budget.jsonl"
    with Budget(path, "manifest", 10) as budget:
        with budget.measurement("bfcl"):
            pass
        used = budget.used_s
    with Budget(path, "manifest", 10) as budget:
        assert budget.used_s == used
        assert budget.remaining_s == 10 - used
        budget.record({"event": "start", "label": "interrupted-swe"})
    with (
        pytest.raises(ValueError, match="terminal budget record"),
        Budget(path, "manifest", 10),
    ):
        pass


@pytest.mark.parametrize("fails", [False, True])
def test_parallel_pair_overlaps_and_charges_one_budget_interval(
    tmp_path, monkeypatch, fails
):
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "pilot": {
            "gpu_by_mode": {"base": 0, "agent_offload": 1},
            "measurement_budget_s": 10,
        }
    }
    write_json(manifest_path, manifest)
    args = SimpleNamespace(
        platform=tmp_path,
        output=tmp_path / "result",
        manifest=manifest_path,
        budget_ledger=tmp_path / "budget.jsonl",
        budget_seconds=10,
        environment_root=None,
        benchmark="bfcl",
        modes=["base", "agent_offload"],
    )
    active = set()
    both = threading.Event()
    intervals = {}

    @contextmanager
    def services(*_):
        yield {
            mode: SimpleNamespace(port=index + 1)
            for index, mode in enumerate(args.modes)
        }

    def run_tasks(
        platform,
        manifest,
        benchmark,
        tasks,
        mode,
        output,
        port,
        environment_root,
        budget_s,
        stop_event,
    ):
        start = time.monotonic()
        active.add(mode)
        if len(active) == 2:
            both.set()
        assert both.wait(5), "Mode executions did not overlap"
        if fails and mode == "base":
            raise RuntimeError("fixture controller failure")
        if fails:
            assert stop_event.wait(5), (
                "Peer controller failure did not cancel this mode"
            )
        else:
            time.sleep(0.05)
        end = time.monotonic()
        intervals[mode] = (start, end)
        write_json(output / "task/terminal.json", {"status": "completed"})
        return {"controller_error": None, "wall_s": end - start}

    monkeypatch.setattr(paired, "paired_services", services)
    monkeypatch.setattr(paired, "run_tasks", run_tasks)
    if fails:
        with pytest.raises(SystemExit, match="incomplete"):
            paired.run_parallel(args, manifest, [{"id": "task"}])
    else:
        paired.run_parallel(args, manifest, [{"id": "task"}])
        assert max(start for start, _ in intervals.values()) < min(
            end for _, end in intervals.values()
        )
    records = [json.loads(line) for line in args.budget_ledger.read_text().splitlines()]
    assert [row["event"] for row in records] == ["budget", "start", "end"]
    execution = json.loads((args.output / "execution.json").read_text())
    assert execution["remaining_budget_s"] == pytest.approx(
        10 - records[-1]["duration_s"]
    )
    for mode in args.modes:
        assert (args.output / mode / "tasks/task/terminal.json").exists()
    assert bool(execution["controller_error"]) == fails


def test_report_includes_failure_quality_and_latency_and_flags_missing_metrics(
    tmp_path,
):
    data = tmp_path / "data.jsonl"
    data.write_text('{"id": "success"}\n{"id": "failed"}\n')
    manifest = tmp_path / "manifest.json"
    write_json(
        manifest,
        {
            "bfcl": {
                "category": {
                    "path": str(data),
                    "sha256": digest(data),
                    "probe_ids": ["success", "failed"],
                }
            },
            "pilot": {"workers": 16, "arrival": "closed_loop_in_manifest_order"},
        },
    )
    root = tmp_path / "experiment"
    write_json(
        root / "experiment.json",
        {
            "manifest": str(manifest),
            "manifest_sha256": digest(manifest),
            "benchmark": "bfcl",
            "subset": "pilot",
            "modes": ["base"],
        },
    )
    for task_id, status, duration in [
        ("success", "completed", 10),
        ("failed", "ReadTimeout", 100),
    ]:
        write_json(
            root / "base/tasks" / task_id / "terminal.json",
            {"status": status, "e2e_s": duration},
        )
    write_json(root / "base/tasks/execution.json", {"wall_s": 100})
    scores = tmp_path / "scores.json"
    write_json(
        scores,
        {
            "experiment": str(root),
            "manifest_sha256": digest(manifest),
            "modes": {
                "base": {
                    "results": [
                        {"task_id": "success", "resolved": True, "status": "graded"},
                        {
                            "task_id": "failed",
                            "resolved": False,
                            "status": "generation_failed",
                        },
                    ]
                }
            },
        },
    )
    report = build_report(root, scores)["modes"]["base"]
    assert report["quality_percent"] == 50
    assert report["all_terminal_e2e_s"]["mean"] == 55
    assert report["observations"]["sampled_cgroup_memory_peak_bytes"] is None
    assert report["observations"]["sampled_gauge_peaks"] == {}


def test_forced_stop_partial_journal_does_not_discard_valid_observations(tmp_path):
    (tmp_path / "journal.jsonl").write_text(
        '{"event": "tool_end", "duration_s": 2.5}\n{"event": "model_resp'
    )
    result = task_observations(tmp_path, {"status": "supervisor_timeout"})
    assert result["tool_times_s"] == [2.5]
    assert result["journal_corrupt_lines"] == [2]


def test_correct_completion_clock_counts_only_graded_success_and_keeps_full_budget():
    tasks = {
        "early_failure": {"resolved": False, "status": "empty_patch", "ended_at": 110},
        "late_arrival_success": {
            "resolved": True,
            "status": "Submitted",
            "arrived_at": 1000,
            "ended_at": 1100,
            "e2e_s": 100,
        },
        "timeout": {"resolved": False, "status": "ReadTimeout", "ended_at": 1600},
    }
    result = completion_progress(tasks, {"started_at": 100, "wall_s": 1500})
    # A 100-second successful task still completed 1000 seconds into the campaign.
    assert [r["resolved"] for r in result["checkpoints"]] == [0, 1, 1]
    assert result["events"] == [
        {
            "task_id": "late_arrival_success",
            "elapsed_s": 1000,
            "cumulative_resolved": 1,
        }
    ]
    assert result["checkpoints"][-1]["resolved_per_gpu_hour"] == 1
    assert (
        result["checkpoints"][-1]["coverage"]
        == "fixed_batch_finished_no_additional_tasks"
    )

    tasks["pending"] = {"resolved": False, "status": "not_started_budget_exhausted"}
    incomplete = completion_progress(tasks, {"started_at": 100, "wall_s": 1500})
    assert [r["resolved"] for r in incomplete["checkpoints"]] == [0, None, None]
    tasks["late_arrival_success"].pop("ended_at")
    unknown = completion_progress(tasks, {"started_at": 100, "wall_s": 1500})
    assert unknown["resolved_without_completion_time"] == ["late_arrival_success"]
    assert unknown["checkpoints"][0]["resolved"] is None


def test_length_diagnostics_keep_timeouts_and_do_not_invent_token_timing():
    records = [
        {
            "event": "model_attempt",
            "lifecycle_id": "a",
            "attempt": 0,
            "input_tokens": 2048,
        },
        {"event": "model_error", "lifecycle_id": "a", "duration_s": 600},
        {
            "event": "model_attempt",
            "lifecycle_id": "b",
            "attempt": 0,
            "input_tokens": 2048,
        },
        {
            "event": "model_response",
            "lifecycle_id": "b",
            "duration_s": 10,
            "response": {"usage": {"completion_tokens": 128}},
        },
        {
            "event": "model_attempt",
            "lifecycle_id": "c",
            "attempt": 0,
            "input_tokens": 2048,
        },
    ]
    result = request_length_diagnostics(
        {"failed_task": {"resolved": False, "requests": request_records(records)}},
        610,
        distribution,
    )
    bins = {r["output_tokens"]: r for r in result["bins"]}
    assert bins["unknown"]["statuses"] == {"model_error": 1, "unacknowledged": 1}
    assert bins["unknown"]["failed_attempt_http_time_s"] == 600
    assert bins["[128,256)"]["input_tokens"] == "[2048,4096)"
    assert bins["[128,256)"]["responses_from_resolved_tasks"] == 0
    assert bins["[128,256)"]["output_tokens_per_run_wall_s"] == pytest.approx(128 / 610)
    assert result["ttft_by_length"] is None
    assert result["decode_time_per_token_by_length"] is None


def test_reuse_fraction_uses_actual_tokens_and_preserves_missing_or_reset_metrics():
    rows = [
        {"name": "vllm:prompt_tokens_total", "labels": {}, "delta": 1000},
        {"name": "vllm:prompt_tokens_cached_total", "labels": {}, "delta": 100},
        {"name": "vllm:prefix_cache_queries_total", "labels": {}, "delta": 100000},
        {"name": "vllm:prefix_cache_hits_total", "labels": {}, "delta": 90000},
        {"name": "vllm:num_preemptions_total", "labels": {}, "delta": None},
    ]
    result = server_diagnostics({"counter_deltas": rows})
    assert result["actual_cached_prompt_token_fraction"] == 0.1
    assert result["cpu_reloaded_prompt_token_fraction"] is None
    assert result["preemptions"] is None
    assert result["aggregate_timers"]["ttft_s"]["mean"] is None
