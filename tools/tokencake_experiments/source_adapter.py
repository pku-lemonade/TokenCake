# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute the frozen workload builder/analyzer in an isolated subprocess."""

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "analyze"])
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    parameters = json.loads(args.input.read_text())
    checkout = args.checkout.resolve()
    if (
        args.command == "freeze"
        and importlib.util.find_spec("vllm.transformers_utils.tokenizer") is None
    ):
        import vllm.tokenizers

        sys.modules["vllm.transformers_utils.tokenizer"] = vllm.tokenizers
    os.chdir(checkout)
    sys.path.insert(0, str(checkout))
    if args.command == "freeze":
        import numpy as np
        import vllm_serving as launcher
        from tools.tokencake_experiments.evidence import make_arrival_trace
        from tools.tokencake_experiments.schema import content_hash

        random.seed(parameters["seed"])
        np.random.seed(parameters["seed"])
        launcher.MODEL_PATH = parameters["model"]
        requests = launcher.generate_requests_app(
            parameters["dataset"],
            parameters["num_requests"],
            "code-paper-pressure",
            {
                "llm_metadata": launcher.LLMCallMetadata(
                    model="gpt-4o-mini", max_new_tokens=500, temperature=0
                )
            },
            workload_source_revision=parameters["workload_revision"],
        )
        contracts = [request.frozen_workload_contract for request in requests]
        result = {
            "workload_contracts": contracts,
            "workload_sha256": content_hash(contracts),
            "arrivals": {
                str(qps): make_arrival_trace(qps, parameters["num_requests"])
                for qps in parameters["qps"]
            },
        }
    else:
        from tools.tokencake_experiments.analysis import (
            summarize_attempt,
            validate_fresh_state,
        )

        state = parameters["state_isolation"]
        if parameters.get("old_protocol"):
            state = validate_fresh_state(
                json.loads(
                    (Path(parameters["case_dir"]) / "state-before.json").read_text()
                )
            )

        result = summarize_attempt(
            Path(parameters["case_dir"]),
            parameters["attempt"],
            parameters["total_e2e_s"],
            parameters["contamination"],
            state,
        )
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
