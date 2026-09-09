# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""mini-swe-agent's text parser, agent loop, and local tool execution."""

import platform
from pathlib import Path

import yaml
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel
from openai.types.chat import ChatCompletion

from .transport import Transport


class BenchmarkTextModel(LitellmTextbasedModel):
    # The shared transport owns every attempt. Disable mini's outer retries as
    # well as avoiding LiteLLM's internal HTTP/retry/cache path.
    abort_exceptions = [Exception, KeyboardInterrupt]

    def __init__(self, transport: Transport, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.transport, self.tokenizer = transport, tokenizer

    def _query(self, messages, **kwargs):
        tokens = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        result = self.transport.complete(
            "chat/completions",
            {"model": self.config.model_name, "messages": messages},
            tokens,
            reusable=True,
        )
        return ChatCompletion.model_validate(result)

    def _calculate_cost(self, response):
        # Local serving has no dollar-based task stop; real usage stays in the
        # unmodified response and the request journal.
        return {"cost": 0.0}


class BenchmarkEnvironment(LocalEnvironment):
    def __init__(self, transport: Transport, **kwargs):
        super().__init__(**kwargs)
        self.transport = transport

    def execute(self, action, cwd="", *, timeout=None):
        remaining = self.transport.context.remaining()
        self.transport.journal.record("shell_command", action=action)
        result = super().execute(
            action, cwd, timeout=min(timeout or self.config.timeout, remaining)
        )
        self.transport.journal.record("shell_result", result=result)
        return result

    def get_template_vars(self, **kwargs):
        # The upstream implementation also exposes all host environment
        # variables. Its selected prompt only needs config and platform facts.
        return self.config.model_dump() | platform.uname()._asdict() | kwargs


class BenchmarkAgent(DefaultAgent):
    def execute_actions(self, message):
        if not message.get("extra", {}).get("actions"):
            return super().execute_actions(message)
        with self.model.transport.tool_window("shell"):
            return super().execute_actions(message)


def run(
    problem: str,
    transport: Transport,
    model: str,
    tokenizer,
    preset: Path,
    task_dir: Path,
    environment: dict[str, str],
    trajectory: Path,
) -> dict:
    config = yaml.safe_load(preset.read_text())
    config["agent"].update(
        cost_limit=0.0,
        step_limit=250,
        wall_time_limit_seconds=3600,
        output_path=trajectory,
    )
    # LocalEnvironment has no /testbed mount. Refer to its current directory
    # relatively so all comparison modes receive the same initial prompt.
    for key in ("system_template", "instance_template"):
        config["agent"][key] = config["agent"][key].replace("/testbed", ".")
    config["environment"].pop("environment_class", None)
    config["environment"].pop("interpreter", None)
    config["environment"].update(cwd=str(task_dir), timeout=60)
    config["environment"]["env"] |= environment
    config["model"].pop("model_class", None)
    config["model"].update(model_name=model, cost_tracking="ignore_errors")
    local_model = BenchmarkTextModel(transport, tokenizer, **config["model"])
    local_env = BenchmarkEnvironment(transport, **config["environment"])
    agent = BenchmarkAgent(local_model, local_env, **config["agent"])
    return agent.run(problem)
