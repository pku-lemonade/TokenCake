# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the frozen launcher using the target environment's tokenizer API."""

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    checkout = Path(sys.argv[1]).resolve()
    launcher = checkout / "vllm_serving.py"
    # The source launcher predates v0.22's tokenizer module move. Keep its
    # algorithm and protocol patch independent of this client-only import alias.
    import vllm.tokenizers

    sys.modules["vllm.transformers_utils.tokenizer"] = vllm.tokenizers
    os.chdir(checkout)
    sys.path.insert(0, str(checkout))
    sys.argv = [str(launcher), *sys.argv[2:]]
    runpy.run_path(str(launcher), run_name="__main__")


if __name__ == "__main__":
    main()
