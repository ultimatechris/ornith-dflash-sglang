"""SGLang serving and eval settings for Ornith-1.5-9B with the DFlash draft.

Read by eval_harness.py. Select it with ORNITH_CONFIG=config_9b (the default).
"""

TARGET_MODEL = "ornith-ai/Ornith-1.5-9B"
DTYPE = "bfloat16"
DFLASH_DRAFT = "z-lab/Qwen3.5-9B-DFlash"

# Flags shared by every configuration; the per-config spec flags are added on
# top. These match the launch command in the README.
SGLANG_COMMON = [
    "--dtype", DTYPE,
    "--reasoning-parser", "qwen3",
    "--mem-fraction-static", "0.85",
    "--context-length", "8192",
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
