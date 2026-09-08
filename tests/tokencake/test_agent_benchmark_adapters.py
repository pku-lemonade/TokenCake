# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration checks against locally installed, frozen official agent code.

These use scripted model responses to validate adapters. They are not benchmark
quality/performance runs and must never be included in experiment summaries.
"""

import json
import os
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from tools.tokencake_experiments.agent_bench.transport import (
    Journal,
    TaskContext,
    Transport,
)


@pytest.fixture
def platform_root():
    value = os.environ.get("TC_AGENT_BENCH_ROOT")
    if not value:
        pytest.skip("Set TC_AGENT_BENCH_ROOT to the prepared official sources")
    assert value is not None
    return Path(value)


@pytest.fixture
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        "/root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct", local_files_only=True
    )


def test_swe_environment_validation_uses_official_test_matching(platform_root):
    # Exercise the actual frozen grader in its independent environment. Its
    # runtime does not need model service/Prometheus dependencies to judge tests.
    program = """
import json
from copy import deepcopy
from tools.tokencake_experiments.agent_bench.environments import reference_checks
task = {'FAIL_TO_PASS': ['test_fixed[param-'],
        'PASS_TO_PASS': ['test_stable[param-']}
original = {'test_output_found': True, 'resolved': False, 'parsed_tests': {
    'test_fixed[param-a]': 'FAILED', 'test_fixed[param-b]': 'FAILED',
    'test_stable[param-a]': 'PASSED', 'test_stable[param-b]': 'XFAIL'}}
reference = {'resolved': True}
results = {'valid': reference_checks(task, original, reference)}
missing = deepcopy(original)
missing['parsed_tests'] = {key: value for key, value in
    missing['parsed_tests'].items() if not key.startswith('test_fixed')}
results['missing'] = reference_checks(task, missing, reference)
skipped = deepcopy(original)
skipped['parsed_tests']['test_stable[param-a]'] = 'SKIPPED'
skipped['parsed_tests']['test_stable[param-b]'] = 'SKIPPED'
results['skipped'] = reference_checks(task, skipped, reference)
conflicting = deepcopy(original)
conflicting['parsed_tests']['test_stable[param-b]'] = 'FAILED'
results['conflicting'] = reference_checks(task, conflicting, reference)
results['unresolved'] = reference_checks(task, original, {'resolved': False})
print(json.dumps(results))
"""
    result = subprocess.run(
        [
            str(platform_root / "environments/grading/.venv/bin/python"),
            "-c",
            program,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    checks = json.loads(result.stdout)
    assert all(checks["valid"].values())
    assert not checks["missing"]["original_f2p_failed"]
    assert not checks["skipped"]["original_p2p_passed"]
    assert not checks["conflicting"]["original_p2p_passed"]
    assert not checks["unresolved"]["reference_resolved"]


def response(text, *, chat=False):
    choice = {"index": 0, "finish_reason": "stop"}
    if chat:
        choice["message"] = {"role": "assistant", "content": text}
    else:
        choice["text"] = text
    return {
        "id": "cmpl-fixture",
        "object": "chat.completion" if chat else "text_completion",
        "created": 1,
        "model": "fixture",
        "choices": [choice],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def test_official_patch_fallback_restores_partial_rejects(platform_root, tmp_path):
    program = r'''
import json, subprocess, sys
from pathlib import Path
from tools.tokencake_experiments.agent_bench.swe_local import apply_model_patch
root = Path(sys.argv[1])
repo = root / "repo"
repo.mkdir()
def git(*args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
git("init")
(repo / "first.txt").write_text("old-first\n")
(repo / "second.txt").write_text(
    "actual-top\nunchanged-top\nold\nunchanged-bottom\nactual-bottom\n")
git("add", ".")
git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
    "commit", "-m", "original")
patch = root / "prediction.patch"
patch.write_text("""diff --git a/first.txt b/first.txt
--- a/first.txt
+++ b/first.txt
@@ -1 +1 @@
-old-first
+new-first
diff --git a/second.txt b/second.txt
--- a/second.txt
+++ b/second.txt
@@ -1,5 +1,5 @@
 expected-top
 unchanged-top
-old
+new
 unchanged-bottom
 expected-bottom
""")
output = root / "output"
output.mkdir()
result = apply_model_patch(repo, {}, patch, output)
result["first"] = (repo / "first.txt").read_text()
result["second"] = (repo / "second.txt").read_text()
result["reject_files"] = [str(path) for path in repo.glob("*.rej")]
print(json.dumps(result))
'''
    completed = subprocess.run(
        [
            str(platform_root / "environments/grading/.venv/bin/python"),
            "-c",
            program,
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = json.loads(completed.stdout)
    assert result["applied"]
    assert [row["exit_code"] for row in result["attempts"]] == [1, 1, 1, 0]
    assert result["attempts"][-1]["command"][0] == "patch"
    assert result["first"] == "new-first\n"
    assert result["second"] == (
        "actual-top\nunchanged-top\nnew\nunchanged-bottom\nactual-bottom\n"
    )
    assert result["reject_files"] == []


@pytest.mark.parametrize("entry_index", [0, 103])
def test_bfcl_official_loop_and_real_tool_state(
    platform_root, tokenizer, tmp_path, entry_index
):
    from bfcl_eval.model_handler.local_inference.qwen import QwenHandler
    from bfcl_eval.utils import (
        make_json_serializable,
        populate_test_cases_with_predefined_functions,
    )
    from openai.types import Completion

    from tools.tokencake_experiments.agent_bench import bfcl
    from tools.tokencake_experiments.agent_bench.inputs import read_jsonl

    data = (
        platform_root
        / "sources/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"
    )
    entry = read_jsonl(data / "BFCL_v4_multi_turn_base.json")[entry_index]
    truth = read_jsonl(data / "possible_answer/BFCL_v4_multi_turn_base.json")[
        entry_index
    ]
    scripted = []
    for calls in truth["ground_truth"]:
        # The reference executor accepts positional arguments; the model's
        # official prompting parser requires named arguments. Preserve the
        # reference action while expressing it in the model's required format.
        calls = [
            call.replace("sort('", "sort(file_name='").replace(
                "get_stock_info('", "get_stock_info(symbol='"
            )
            for call in calls
        ]
        scripted.extend(["[" + ", ".join(calls) + "]", "[]"])

    official = QwenHandler("reference-only", 0.0, "fixture", False)
    official.tokenizer = tokenizer
    official.max_context_length = 32768
    official.model_path_or_id = "fixture"
    expected_requests = []
    script = iter(scripted)

    def official_complete(**kwargs):
        expected_requests.append(kwargs)
        return Completion.model_validate(response(next(script)))

    with patch.object(official.client.completions, "create", official_complete):
        reference, reference_meta = official.inference(
            populate_test_cases_with_predefined_functions([deepcopy(entry)])[0],
            True,
            False,
        )
    official.client.close()

    requests, events = [], []
    script = iter(scripted)

    def http_handler(request):
        body = json.loads(request.content)
        if "event" in body:
            events.append(body["event"])
            return httpx.Response(200, json=body | {"disposition": "applied"})
        requests.append(body)
        return httpx.Response(200, json=response(next(script)))

    journal = Journal(tmp_path / "journal.jsonl")
    context = TaskContext(
        entry["id"], "bfcl", "agent_offload", time.time(), 0.0, time.time() + 60
    )
    transport = Transport(
        "http://local/v1",
        context,
        journal,
        client=httpx.Client(transport=httpx.MockTransport(http_handler)),
    )
    try:
        result = bfcl.run(entry, transport, "fixture", tokenizer)
    finally:
        transport.close()
        journal.close()
    assert result["result"] == reference
    # JSON converts tuples and integer object keys as well as BFCL's sets.
    persisted = json.loads(json.dumps(result))
    assert persisted["inference_log"] == json.loads(
        json.dumps(make_json_serializable(reference_meta["inference_log"]))
    )
    assert [r["prompt"] for r in requests] == [r["prompt"] for r in expected_requests]
    assert events == ["stall_started", "stall_finished"] * len(truth["ground_truth"])
    assert bfcl.grade(entry, truth, result)["valid"]
    assert not bfcl.grade(entry, truth, {"result": []})["valid"]


def test_mini_official_parser_submission_and_actual_shell(
    platform_root, tokenizer, tmp_path
):
    from tools.tokencake_experiments.agent_bench import mini

    script = iter(
        [
            (
                "THOUGHT: Run the tool.\n<mswea_bash_command>"
                "printf result > computed.txt</mswea_bash_command>"
            ),
            (
                "THOUGHT: Submit.\n<mswea_bash_command>"
                "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat computed.txt"
                "</mswea_bash_command>"
            ),
        ]
    )
    observed = []

    def handler(request):
        body = json.loads(request.content)
        observed.append(body.get("event", "model"))
        if "event" in body:
            return httpx.Response(200, json=body | {"disposition": "applied"})
        assert "tools" not in body
        return httpx.Response(200, json=response(next(script), chat=True))

    journal = Journal(tmp_path / "journal.jsonl")
    context = TaskContext(
        "fixture", "swe_coder", "agent_offload", time.time(), 0.0, time.time() + 60
    )
    transport = Transport(
        "http://local/v1",
        context,
        journal,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    try:
        result = mini.run(
            "Produce the requested file.",
            transport,
            "fixture",
            tokenizer,
            platform_root
            / "sources/mini-swe-agent/src/minisweagent/config/benchmarks"
            / "swebench_xml.yaml",
            tmp_path,
            {},
            tmp_path / "trajectory.json",
        )
    finally:
        transport.close()
        journal.close()
    assert result["exit_status"] == "Submitted"
    assert result["submission"] == "result"
    assert (tmp_path / "computed.txt").read_text() == "result"
    assert observed == ["model", "stall_started", "stall_finished"] * 2
