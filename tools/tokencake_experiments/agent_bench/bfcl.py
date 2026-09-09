# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve BFCL's Qwen prompting loop and official multi-turn checker."""

import time
from copy import deepcopy
from unittest.mock import patch

from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
    is_empty_execute_response,
)
from bfcl_eval.model_handler.local_inference.qwen import QwenHandler
from bfcl_eval.utils import (
    make_json_serializable,
    populate_test_cases_with_predefined_functions,
)
from openai.types import Completion
from overrides import override

from .transport import Transport


class BenchmarkQwenHandler(QwenHandler):
    def __init__(self, transport: Transport, model: str, tokenizer):
        super().__init__(model, 0.0, "TokenCake-Qwen2.5-14B", False)
        self.client.close()  # All HTTP attempts use our explicit retry policy.
        self.transport = transport
        self.tokenizer = tokenizer
        self.model_path_or_id = model
        self.max_context_length = transport.max_context

    @override
    def _query_prompting(self, inference_data: dict):
        prompt = self._format_prompt(
            inference_data["message"], inference_data["function"]
        )
        inference_data["inference_input_log"] = {"formatted_prompt": prompt}
        # Match the server's Completions tokenization, retaining the official
        # manually formatted Qwen prompt. Never shorten the conversation.
        tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        payload = {"model": self.model_path_or_id, "prompt": prompt}
        for field in ("stop_token_ids", "skip_special_tokens"):
            if hasattr(self, field):
                payload[field] = getattr(self, field)
        started = time.monotonic()
        result = self.transport.complete("completions", payload, tokens, reusable=True)
        return Completion.model_validate(result), time.monotonic() - started


def run(entry: dict, transport: Transport, model: str, tokenizer) -> dict:
    """Called once per dedicated process: BFCL keeps tool objects in globals."""
    entry = populate_test_cases_with_predefined_functions([deepcopy(entry)])[0]
    handler = BenchmarkQwenHandler(transport, model, tokenizer)

    def execute(calls, *args, **kwargs):
        if not calls:  # The official runner initializes state before querying.
            return execute_multi_turn_func_call(calls, *args, **kwargs)
        with transport.tool_window("bfcl_functions"):
            return execute_multi_turn_func_call(calls, *args, **kwargs)

    # Patch only the inference module's reference. Evaluation uses a separate
    # process and the original executor. The official @final loop is unchanged.
    with patch(
        "bfcl_eval.model_handler.base_handler.execute_multi_turn_func_call", execute
    ):
        response, metadata = handler.inference(
            entry, include_input_log=True, exclude_state_log=False
        )
    # Official tool state logs include sets and other Python objects. Use BFCL's
    # result-file conversion before passing them to our durable JSON writer.
    return make_json_serializable({"id": entry["id"], "result": response, **metadata})


def grade(entry: dict, answer: dict, prediction: dict) -> dict:
    """Adapt result loading only; state/response checks are BFCL's own functions.

    Turn count and decoding follow _evaluate_single_multi_turn_entry from the
    frozen official eval_runner. Importing that entire CLI eagerly loads all
    vendor handlers and unrelated agentic benchmark dependencies.
    """
    model_result = prediction.get("result")
    truth = answer["ground_truth"]
    if not isinstance(model_result, list):
        return {"valid": False, "error_type": "multi_turn:inference_error"}
    if len(model_result) != len(truth):
        return {"valid": False, "error_type": "multi_turn:force_terminated"}
    handler = QwenHandler("Qwen/Qwen2.5-14B-Instruct", 0.0, "TokenCake", False)
    handler.client.close()
    decoded = []
    for turn in model_result:
        steps = []
        for response in turn:
            try:
                calls = handler.decode_execute(response, has_tool_call_tag=False)
                if not is_empty_execute_response(calls):
                    steps.append(calls)
            except Exception:
                # Official scoring ignores prose and malformed function calls.
                continue
        decoded.append(steps)
    return multi_turn_checker(
        decoded, truth, deepcopy(entry), entry["id"].rsplit("_", 1)[0], "TokenCake"
    )
