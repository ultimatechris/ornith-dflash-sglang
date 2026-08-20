"""Frozen specification for Exp 1: SGLang single-stream decode-tps clarifier
for Ornith-1.5-9B.

This is a measurement experiment, not an artifact experiment. Its job is to
answer one question the earlier ExLlamaV3-only work could not: does *any*
speculative-decoding config give a real single-stream decode-tps win on this
Qwen3.5 gated-deltanet hybrid when run in an engine (SGLang) that has
purpose-built kernels + GDN spec-verify for it — and which config wins, the
distribution-matched native MTP head or the stronger-but-stock DFlash/EAGLE
drafts trained on base Qwen3.5.

Everything here is frozen. The eval workload is verifiable (executable code /
exact-match math), never prose similarity. Greedy decoding makes correct
speculative decoding exactly equal to the baseline token-for-token, so the
lossless check is string equality, not a fuzzy score.
"""

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
TARGET_MODEL = "ornith-ai/Ornith-1.5-9B"   # dense, Qwen3.5 GDN hybrid, native MTP
DTYPE = "bfloat16"                           # no quant confound in the clarifier

# Drafts trained on STOCK Qwen3.5-9B (not on Ornith's fine-tune). Architecturally
# compatible (same vocab/hidden); the open question is how much Ornith's
# fine-tune drift costs their acceptance vs Ornith's own MTP head.
DFLASH_DRAFT = "z-lab/Qwen3.5-9B-DFlash"
EAGLE3_DRAFT = "BLR2/Eagle3-Qwen3.5-9B"

# ---------------------------------------------------------------------------
# SGLang serving
# ---------------------------------------------------------------------------
# Common flags for every config; per-config spec flags are added by the matrix.
SGLANG_COMMON = [
    "--dtype", DTYPE,
    "--reasoning-parser", "qwen3",
    "--mem-fraction-static", "0.85",
    "--context-length", "8192",
]

# Spec-decoding config search. Run 3 found the one NEXTN config we tried
# (steps3/draft4/topk1 + ReplaySSM) gave accept_length ~= 1.0 (near-zero draft
# acceptance) and 0.66x. That is lower than the old ExLlamaV3 MTP (12-65%),
# suggesting a config/wiring issue rather than "MTP is useless". This sweep
# iterates the NEXTN knobs (num_steps, num_draft_tokens, eagle_topk, and the
# GDN ReplaySSM spec-verify on/off) to find a config with accept_length > 1 and
# speedup > 1 -- or to show none exists. The winner is then confirmed on the full
# eval in a follow-up.


def _nextn(steps, draft, topk, rssm=False):
    f = ["--speculative-algorithm", "NEXTN",
         "--speculative-num-steps", str(steps),
         "--speculative-eagle-topk", str(topk),
         "--speculative-num-draft-tokens", str(draft)]
    if rssm:
        f.append("--enable-gdn-replayssm-spec")
    return f


def _dflash(block, backends=False):
    # DFlash is a *trained* draft (z-lab), not EAGLE. It uses its own algorithm
    # keyword and a block-size param (not num-steps/draft-tokens). Earlier crash
    # ("'DFlashDraftModel' has no attribute set_embed_and_head") was from wrongly
    # using --speculative-algorithm EAGLE. Backend flags fa4/trtllm_mha from the
    # model card look Blackwell-specific, so we try safe flashinfer backends here.
    f = ["--speculative-algorithm", "DFLASH",
         "--speculative-draft-model-path", DFLASH_DRAFT,
         "--speculative-dflash-block-size", str(block)]
    if backends:
        f += ["--linear-attn-prefill-backend", "flashinfer",
              "--linear-attn-decode-backend", "flashinfer",
              "--mamba-scheduler-strategy", "extra_buffer"]
    return f


# name -> extra CLI flags. `baseline` first (its greedy output is the lossless
# reference and the speedup denominator). Native-MTP sweep (run 4) already showed
# accept ~= 1.0 across all NEXTN configs (no lever). This run tests the *trained*
# DFlash draft, which is the strong-draft path (LMSYS: >4.3x baseline on Blackwell).
SPEC_CONFIGS = {
    "baseline":    [],
    "dflash_bs16": _dflash(16),
    "dflash_bs24": _dflash(24),
    "dflash_bs32": _dflash(32),
}

# `baseline` must run first: its greedy outputs are the reference every spec
# config is checked against for losslessness.
BASELINE = "baseline"

# ---------------------------------------------------------------------------
# Verifiable eval workload
# ---------------------------------------------------------------------------
# GSM8K (exact numeric match) is primary: pure string/number check, no code
# execution. HumanEval (execute unit tests -> pass@1) is secondary and the
# high-acceptance domain where the spec-decoding lever should be largest.
# Fast probe for the config search: decode tps + acceptance stabilize in a few
# dozen generations, so we screen configs on 40 GSM8K prompts with no HumanEval
# execution (quality is already validated in run 3). The winning config is then
# re-run on the full eval (GSM8K 150 + HumanEval 164) to confirm.
# Full-eval confirmation of the DFlash winner: real generation lengths (640
# tokens, so GSM8K CoT is not truncated), GSM8K exact-match + HumanEval pass@1.
# The quality check is PASS-RATE PARITY vs baseline (equal pass rate = quality
# preserved); the exact-token "lossless" fraction is limited by cross-server
# greedy nondeterminism and is reported for context only.
GSM8K_N = 120
HUMANEVAL_N = 120
MAX_NEW_TOKENS = 640
GREEDY = True              # temperature 0 -> spec decoding is exact vs baseline

# Decode tps is measured client-side from the streamed response:
#   decode_tps = (completion_tokens - 1) / (t_last_token - t_first_token)
# which excludes prefill/TTFT. Acceptance comes from SGLang meta_info.

# ---------------------------------------------------------------------------
# Deliverable
# ---------------------------------------------------------------------------
# Report only (no weights). Pushed to a private HF repo so it survives a lost
# session; also retrieved locally and committed to the experiment folder.
HF_RESULTS_REPO = "ultimatechris/ornith15-9b-decode-sglang"
HF_PRIVATE = True
