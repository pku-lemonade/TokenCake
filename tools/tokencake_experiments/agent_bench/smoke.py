# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real serving protocol validation, explicitly excluded from benchmark scores."""

import argparse
import subprocess
import time
from pathlib import Path

from transformers import AutoTokenizer

from tools.tokencake_experiments.campaign import (
    add_settings_arguments,
    settings_from_args,
)
from tools.tokencake_experiments.provenance import resolve_model

from .service import Service
from .transport import Journal, TaskContext, Transport, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    add_settings_arguments(parser)
    args = parser.parse_args()
    settings = settings_from_args(args)
    model = str(resolve_model(settings.model))
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(model)
    results = []
    for mode in ("agent_offload", "base"):
        with Service(
            args.platform,
            args.output / mode / "service",
            mode,
            gpu=settings.gpus[0],
            model=model,
            settings=settings.payload() | {"cpu_kv_gib": settings.cpu_offload_gib},
        ) as server:
            context = TaskContext(
                "protocol-smoke", "swe_coder", mode, time.time(), 0.0, time.time() + 600
            )
            journal = Journal(args.output / mode / "journal.jsonl")
            transport = Transport(
                f"http://127.0.0.1:{server.port}/v1",
                context,
                journal,
                max_context=settings.max_model_len,
            )
            messages = [
                {
                    "role": "user",
                    "content": "What is 2 + 2? Answer with only the number.",
                }
            ]
            outputs = []
            try:
                for step in range(2):
                    tokens = tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True
                    )
                    result = transport.complete(
                        "chat/completions",
                        {"model": model, "messages": messages, "max_tokens": 32},
                        tokens,
                        reusable=True,
                    )
                    if result["usage"]["prompt_tokens"] != len(tokens):
                        raise ValueError(
                            "Local and server prompt token counts disagree"
                        )
                    text = result["choices"][0]["message"]["content"]
                    outputs.append(text)
                    messages.append({"role": "assistant", "content": text})
                    with transport.tool_window("shell"):
                        subprocess.run(["true"], check=True)
                    messages.append(
                        {
                            "role": "user",
                            "content": "The command succeeded. Repeat the number.",
                        }
                    )
                results.append({"mode": mode, "outputs": outputs})
            finally:
                transport.close()
                journal.close()
    write_json(
        args.output / "result.json",
        {
            "role": "protocol_validation_not_benchmark",
            "results": results,
            "same_output": results[0]["outputs"] == results[1]["outputs"],
        },
    )


if __name__ == "__main__":
    main()
