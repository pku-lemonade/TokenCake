# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Versioned JSON workloads with explicit dependencies and conversation templates."""

import hashlib
import json
import math
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Tool(Record):
    kind: str = Field(min_length=1)
    duration_s: float = Field(ge=0)
    result: str
    preserve_generation: bool = False


class Node(Record):
    name: str = Field(min_length=1)
    type: str = Field(min_length=1)
    kind: Literal["input", "collect", "llm"]
    predecessors: list[str]
    template: str | None = None
    composition: Literal["prepend", "prepend_shared", "append_shared", "same_role"]
    max_tokens: int = Field(ge=0)
    metadata: dict[str, Any]
    tool: Tool | None = None

    @model_validator(mode="after")
    def execution_kind(self) -> Self:
        if self.kind == "llm" and (not self.template or self.max_tokens == 0):
            raise ValueError("Model nodes need a template and positive output budget")
        if self.kind != "llm" and (self.max_tokens or self.tool):
            raise ValueError("Local nodes cannot generate tokens or execute tools")
        if len(set(self.predecessors)) != len(self.predecessors):
            raise ValueError("Duplicate dependency")
        return self


class Application(Record):
    id: str = Field(min_length=1)
    initial_input: str = Field(min_length=1)
    context_token_count: int = Field(gt=0)
    context_sources: list[str]
    context_source_revision: str
    tool_results: dict[str, str] = Field(default_factory=dict)


class Dataset(Record):
    schema_version: Literal[1]
    name: str
    profile: str
    application_class: str
    application_prompt: str
    max_depth: int = Field(gt=0)
    input_composition: str | None
    tool_instruction_max_tokens: int | None
    provenance: dict[str, Any]
    templates: dict[str, str]
    nodes: list[Node]
    applications: list[Application] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        names = [node.name for node in self.nodes]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate node name")
        if len({app.id for app in self.applications}) != len(self.applications):
            raise ValueError("Duplicate application id")
        inputs = [node for node in self.nodes if node.kind == "input"]
        if len(inputs) != 1 or inputs[0].predecessors:
            raise ValueError("Exactly one input node without dependencies is required")
        known = set(names)
        for node in self.nodes:
            if not set(node.predecessors) <= known:
                raise ValueError(f"Unknown dependency for {node.name}")
            if node.kind != "input" and not node.predecessors:
                raise ValueError(f"Disconnected node: {node.name}")
            if node.template is not None and node.template not in self.templates:
                raise ValueError(f"Unknown template for {node.name}")
        try:
            tuple(TopologicalSorter(self.dependencies).static_order())
        except CycleError as exc:
            raise ValueError("Cycle in dataset DAG") from exc
        tools = {node.name for node in self.nodes if node.tool is not None}
        for app in self.applications:
            if not set(app.tool_results) <= tools:
                raise ValueError(f"Unknown tool result in application {app.id}")
        return self

    @property
    def dependencies(self) -> dict[str, list[str]]:
        return {node.name: node.predecessors for node in self.nodes}

    def contract(self, app: Application) -> dict:
        contract = {
            "application": self.application_class,
            "task": self.name,
            "application_prompt_sha256": text_hash(self.application_prompt),
            "dataset_prompt_sha256": text_hash(app.initial_input),
            "expected_dag_nodes": sorted(
                (
                    {
                        "name": node.name,
                        "type": node.type,
                        "is_mcp": node.tool is not None,
                    }
                    for node in self.nodes
                ),
                key=lambda node: node["name"],
            ),
        }
        if self.input_composition is not None:
            contract["input_composition"] = self.input_composition
        if self.tool_instruction_max_tokens is not None:
            contract["tool_instruction_max_tokens"] = self.tool_instruction_max_tokens
        return contract

    def freeze(self, qps_values: list[float]) -> dict:
        if not qps_values or any(
            not math.isfinite(qps) or qps <= 0 for qps in qps_values
        ):
            raise ValueError("Arrival QPS must be positive and finite")
        contracts = [self.contract(app) for app in self.applications]
        return {
            "workload_contracts": contracts,
            "workload_sha256": text_hash(
                json.dumps(contracts, sort_keys=True, separators=(",", ":"))
            ),
            "arrivals": {
                str(qps): [index / qps for index in range(len(contracts))]
                for qps in qps_values
            },
        }


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_dataset(path: Path) -> Dataset:
    return Dataset.model_validate_json(path.read_bytes())


def compose_input(
    dataset: Dataset, node: Node, outputs: dict[str, list[str]]
) -> list[str]:
    chains = [outputs[name] for name in node.predecessors]
    template = dataset.templates[node.template] if node.template is not None else ""
    if node.kind == "collect" or node.composition == "prepend":
        return [template, *(chunk for chain in chains for chunk in chain)]
    if node.composition == "same_role":
        if len(chains) == 1 and chains[0] and chains[0][0] == template:
            return list(chains[0])
        return [template, *(chunk for chain in chains for chunk in chain)]
    common = min((len(chain) for chain in chains), default=0)
    for index in range(common):
        if any(chain[index] != chains[0][index] for chain in chains[1:]):
            common = index
            break
    chunks = list(chains[0][:common]) if chains else []
    for chain in chains:
        chunks.extend(chain[common:])
    return (
        chunks + [template]
        if node.composition == "append_shared"
        else [template] + chunks
    )


def compose_output(generated: str, tool_result: str, preserve_generation: bool) -> str:
    if not preserve_generation:
        return tool_result
    return generated + ("" if generated.endswith("\n") else "\n") + tool_result
