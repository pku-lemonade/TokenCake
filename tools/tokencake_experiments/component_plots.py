# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot the complete component matrix with explicit per-launch observations."""

import argparse
import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.tokencake_experiments.component_matrix import (
    LABELS as MODE_LABELS,
)
from tools.tokencake_experiments.component_matrix import (
    LOADS,
    MODES,
    group_runs,
)

LABELS = tuple(MODE_LABELS[mode] for mode in MODES)
COLORS = ("#555555", "#167eab", "#d78a22", "#c74449")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.data.read_text())
    groups = group_runs(rows)
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 180,
        }
    )

    def save(fig, name):
        fig.savefig(args.output / f"{name}.png", bbox_inches="tight")
        fig.savefig(args.output / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    def plot(ax, getter, title, ylabel):
        for mode, label, color in zip(MODES, LABELS, COLORS):
            values = [[getter(row) for row in groups[mode, qps]] for qps in LOADS]
            centers = np.array([median(value) for value in values])
            errors = np.array(
                [
                    [centers[i] - min(value) for i, value in enumerate(values)],
                    [max(value) - centers[i] for i, value in enumerate(values)],
                ]
            )
            ax.errorbar(
                LOADS,
                centers,
                yerr=errors,
                marker="o",
                markersize=4,
                linewidth=1.7,
                capsize=3,
                color=color,
                label=label,
            )
        ax.set_xscale("log")
        ax.set_xticks(LOADS, [str(value) for value in LOADS])
        ax.set_xlabel("Offered application QPS")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(bottom=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for ax, (field, title) in zip(
        axes,
        (
            ("total_e2e_s", "24-DAG batch E2E"),
            ("average_app_latency_s", "Mean application latency"),
            ("p95_app_latency_s", "P95 application latency"),
        ),
    ):
        plot(ax, lambda r, field=field: r["performance"][field], title, "Seconds")
    axes[0].legend(fontsize=9)
    save(fig, "latency")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    plot(
        axes[0],
        lambda r: r["token_counts"]["generated"] / r["performance"]["total_e2e_s"],
        "Completed output throughput",
        "Output tokens / second",
    )
    plot(
        axes[1],
        lambda r: 100 * r["timeseries"]["kv_cache_usage_perc"]["mean"],
        "Time-weighted KV occupancy",
        "Percent of KV blocks",
    )
    plot(
        axes[2],
        lambda r: r["hardware_timeseries"]["utilization.gpu"]["mean"],
        "Time-weighted GPU activity",
        "Percent",
    )
    axes[1].set_ylim(0, 101)
    axes[2].set_ylim(0, 101)
    axes[0].legend(fontsize=9)
    save(fig, "throughput-memory")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for mode, label, color in zip(MODES[1:], LABELS[1:], COLORS[1:]):
        improvements = []
        for qps in LOADS:
            base = median(
                r["performance"]["total_e2e_s"] for r in groups["native", qps]
            )
            value = median(r["performance"]["total_e2e_s"] for r in groups[mode, qps])
            improvements.append(100 * (1 - value / base))
        axes[0].plot(LOADS, improvements, marker="o", color=color, label=label)
    axes[0].axhline(25, color="#777777", linestyle=":", label="25% reference")
    axes[0].set_ylabel("E2E reduction from base (%)")
    axes[0].legend(fontsize=9)
    interactions = []
    for qps in LOADS:
        t = {
            mode: median(r["performance"]["total_e2e_s"] for r in groups[mode, qps])
            for mode in MODES
        }
        interactions.append(
            t["agent"] + t["offload"] - t["native"] - t["offload-agent"]
        )
    axes[1].plot(LOADS, interactions, marker="s", color="#43805c")
    axes[1].axhline(0, color="#777777", linestyle=":")
    axes[1].set_ylabel("Interaction: Ta + To - Tb - Tf (s)")
    axes[1].set_title("Positive values indicate added combined benefit")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(LOADS, [str(value) for value in LOADS])
        ax.set_xlabel("Offered application QPS")
        ax.grid(axis="y", alpha=0.2)
    save(fig, "component-effects")

    fig, axes = plt.subplots(1, 5, figsize=(17, 4), layout="constrained", sharey=True)
    sources = ("local_compute", "local_cache_hit", "external_kv_transfer")
    source_colors = ("#777777", "#398557", "#d6a137")
    for ax, qps in zip(axes, LOADS):
        bottom = np.zeros(4)
        for source, color in zip(sources, source_colors):
            values = np.array(
                [
                    median(
                        row["first_prefill_sources"][source] / 1e6
                        for row in groups[mode, qps]
                    )
                    for mode in MODES
                ]
            )
            ax.bar(LABELS, values, bottom=bottom, label=source, color=color)
            bottom += values
        ax.set_title(f"QPS {qps}")
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("First-prefill input tokens (millions)")
    axes[-1].legend(fontsize=8)
    save(fig, "prefill-sources")

    fig, axes = plt.subplots(1, 5, figsize=(17, 4), layout="constrained", sharey=True)
    for ax, qps in zip(axes, LOADS):
        for mode, label, color in zip(MODES, LABELS, COLORS):
            representative = sorted(
                groups[mode, qps], key=lambda r: r["performance"]["total_e2e_s"]
            )[len(groups[mode, qps]) // 2]
            values = sorted(representative["application_latencies_s"])
            ax.step(
                values,
                np.arange(1, len(values) + 1) / len(values),
                where="post",
                label=label,
                color=color,
            )
        ax.set_title(f"QPS {qps}")
        ax.set_xlabel("Application latency (s)")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Empirical CDF (median-E2E launch)")
    axes[-1].legend(fontsize=8)
    save(fig, "application-cdf")


if __name__ == "__main__":
    main()
