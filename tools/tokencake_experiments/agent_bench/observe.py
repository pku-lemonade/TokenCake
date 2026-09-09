# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observe GPU competition and the actual cgroup memory/CPU pressure."""

import os
from pathlib import Path

import psutil

from tools.tokencake_experiments.runtime import Monitor


class PressureMonitor(Monitor):
    def sample(self) -> dict:
        sample = super().sample()
        cgroup = Path("/sys/fs/cgroup")
        values = {}
        for name in (
            "memory.current",
            "memory.max",
            "memory.peak",
            "memory.events",
            "memory.stat",
            "memory.pressure",
            "memory.swap.current",
            "memory.swap.max",
            "cpu.max",
            "cpu.stat",
            "cpu.pressure",
            "pids.current",
            "pids.max",
        ):
            path = cgroup / name
            values[name] = path.read_text().strip() if path.exists() else None
        sample.update(
            {
                "cgroup": values,
                "host_memory": psutil.virtual_memory()._asdict(),
                "host_swap": psutil.swap_memory()._asdict(),
                "host_load_average": os.getloadavg(),
            }
        )
        return sample
