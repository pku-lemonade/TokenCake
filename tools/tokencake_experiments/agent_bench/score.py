# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Score saved predictions in separate processes with the official checkers."""

import argparse
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from .inputs import ROOT, digest, read_jsonl
from .transport import write_json


def score_one(spec: dict, output: Path) -> dict:
    task, terminal = spec["task"], spec["terminal"]
    result = {"resolved": False, "run_status": terminal["status"]}
    if spec["benchmark"] == "bfcl":
        if terminal["status"] != "completed":
            return result | {"status": "generation_failed"}
        from .bfcl import grade

        report = grade(task, spec["answer"], terminal["prediction"])
        return result | {
            "status": "graded",
            "resolved": bool(report["valid"]),
            "official_report": report,
        }
    prediction = terminal.get("prediction", {})
    patch = prediction.get("submission", "")
    if terminal["status"] != "Submitted":
        return result | {"status": "generation_failed"}
    if not patch.strip():
        return result | {"status": "empty_patch"}
    from .swe_local import grade

    report = grade(
        Path(spec["platform"]),
        task,
        Path(spec["environment"]),
        {
            "instance_id": task["instance_id"],
            "model_patch": patch,
            "model_name_or_path": spec["mode"],
        },
        output / "grading",
    )
    return result | report


def score_experiment(
    platform: Path, experiment: Path, output: Path, environment_root: Path | None
):
    from .experiment import tasks_for, terminate
    from .service import worker_environment

    config = json.loads((experiment / "experiment.json").read_text())
    manifest_path = Path(config["manifest"])
    if digest(manifest_path) != config["manifest_sha256"]:
        raise ValueError("Experiment manifest changed")
    manifest = json.loads(manifest_path.read_text())
    benchmark = config["benchmark"]
    if benchmark == "swe" and environment_root is None:
        raise ValueError("SWE grading requires a separate prepared environment root")
    tasks = tasks_for(manifest, benchmark, config["subset"])
    answers = {}
    if benchmark == "bfcl":
        for category in manifest["bfcl"].values():
            path = Path(category["answers_path"])
            if digest(path) != category["answers_sha256"]:
                raise ValueError("BFCL answers changed after freezing")
            answers.update({entry["id"]: entry for entry in read_jsonl(path)})
    output.mkdir(parents=True, exist_ok=False)
    sources = []
    for path in Path(__file__).parent.glob("*.py"):
        destination = output / "adapter-source" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        sources.append(
            {"path": str(path), "copy": str(destination), "sha256": digest(path)}
        )
    write_json(
        output / "grading-implementation.json",
        {
            "adapter_sources": sources,
            "official_grader": manifest["repositories"]["SWE-bench"],
            "official_bfcl": manifest["repositories"]["gorilla"],
            "predictions": str(experiment),
        },
    )
    summaries = {}
    for mode in config["modes"]:
        results = []
        for task in tasks:
            task_id = task.get("instance_id", task.get("id"))
            task_output = output / mode / task_id
            path = experiment / mode / "tasks" / task_id / "terminal.json"
            terminal = (
                json.loads(path.read_text())
                if path.exists()
                else {"status": "terminal_missing"}
            )
            spec = {
                "platform": str(platform),
                "task": task,
                "terminal": terminal,
                "benchmark": benchmark,
                "mode": mode,
            }
            if benchmark == "bfcl":
                spec["answer"] = answers[task_id]
                python = platform / ".venv/bin/python"
            else:
                assert environment_root is not None
                directory = environment_root / task_id
                agent_root = Path(config["environment_root"])
                if directory.resolve() == (agent_root / mode / task_id).resolve():
                    raise ValueError("Agent and grading directories must be separate")
                spec["environment"] = str(directory)
                python = platform / "environments/grading/.venv/bin/python"
            write_json(task_output / "spec.json", spec)
            command = [
                str(python),
                "-m",
                "tools.tokencake_experiments.agent_bench.score",
                "--one",
                str(task_output / "spec.json"),
                "--output",
                str(task_output),
            ]
            try:
                with (task_output / "scorer.log").open("x") as stream:
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=worker_environment(platform, task_output),
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    try:
                        code = process.wait(timeout=3900)
                    except BaseException:
                        terminate(process)
                        raise
                    if code:
                        raise RuntimeError(f"Scorer exited with status {code}")
                result = json.loads((task_output / "score.json").read_text())
            except Exception as exc:
                result = {
                    "status": "scorer_error",
                    "resolved": False,
                    "error": str(exc),
                    "run_status": terminal["status"],
                }
                write_json(task_output / "score-error.json", result)
            results.append({"task_id": task_id, **result})
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "task_id": task_id,
                        "score_status": result["status"],
                        "resolved": result["resolved"],
                    }
                ),
                flush=True,
            )
        summaries[mode] = {
            "planned": len(tasks),
            "recorded": len(results),
            "resolved": sum(bool(row["resolved"]) for row in results),
            "statuses": dict(Counter(row["status"] for row in results)),
            "results": results,
        }
    write_json(
        output / "scores.json",
        {
            "experiment": str(experiment),
            "manifest_sha256": config["manifest_sha256"],
            "benchmark": benchmark,
            "subset": config["subset"],
            "modes": summaries,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one", type=Path)
    parser.add_argument("--platform", type=Path)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--environment-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.one:
        spec = json.loads(args.one.read_text())
        try:
            result = score_one(spec, args.output)
        except Exception as exc:
            result = {
                "status": "scorer_error",
                "resolved": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        write_json(args.output / "score.json", result)
    else:
        if not args.platform or not args.experiment:
            parser.error("Provide --platform and --experiment")
        score_experiment(
            args.platform, args.experiment, args.output, args.environment_root
        )


if __name__ == "__main__":
    main()
