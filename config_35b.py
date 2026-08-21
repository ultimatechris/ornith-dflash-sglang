"""SGLang serving and eval settings for Ornith-1.5-35B-A3B with the DFlash draft.

Read by eval_harness.py. Select it with ORNITH_CONFIG=config_35b. Set NUM_GPUS
to the tensor-parallel size: 2 for two 80 GB cards, or 1 for a single card with
about 90 GB or more.
"""

import os

TARGET_MODEL = "ornith-ai/Ornith-1.5-35B-A3B"
DTYPE = "bfloat16"
DFLASH_DRAFT = "z-lab/Qwen3.5-35B-A3B-DFlash"

TP = int(os.environ.get("NUM_GPUS", "2"))

# The last four flags matter on this model. --max-total-tokens and
# --max-running-requests keep the KV pool and the mamba state cache small enough
# that the CUDA graphs still fit; --mm-feature-transport cpu avoids the
# pidfd_getfd syscall some hosts block for the vision feature transport (this
# recipe sends only text).
SGLANG_COMMON = [
    "--dtype", DTYPE,
    "--reasoning-parser", "qwen3",
    "--mem-fraction-static", "0.80",
    "--max-total-tokens", "40960",
    "--max-running-requests", "4",
    "--mm-feature-transport", "cpu",
    "--context-length", "4096",
    "--tp", str(TP),
]


def _dflash(block):
    return ["--speculative-algorithm", "DFLASH",
            "--speculative-draft-model-path", DFLASH_DRAFT,
            "--speculative-dflash-block-size", str(block)]


# baseline runs first: it is the speedup denominator.
SPEC_CONFIGS = {
    "baseline":    [],
    "dflash_bs16": _dflash(16),
}
BASELINE = "baseline"

# Eval workload: GSM8K exact-match and HumanEval executed pass@1, greedy, 640
# tokens per generation.
GSM8K_N = 120
HUMANEVAL_N = 120
MAX_NEW_TOKENS = 640
GREEDY = True
