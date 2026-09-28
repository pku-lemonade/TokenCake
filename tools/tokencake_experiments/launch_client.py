# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the repository-local client captured during experiment preparation."""

import os
import runpy
import sys
from pathlib import Path


def main():
    checkout = Path(sys.argv[1]).resolve()
    launcher = checkout / "dataset_client.py"
    os.chdir(checkout)
    sys.path.insert(0, str(checkout))
    sys.argv = [str(launcher), *sys.argv[2:]]
    runpy.run_path(str(launcher), run_name="__main__")


if __name__ == "__main__":
    main()
