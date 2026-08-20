# Exp 1 rules — SGLang decode-tps clarifier for Ornith-1.5-9B

## Question

Does *any* speculative-decoding configuration give a real **single-stream
decode tok/s** win on Ornith-1.5-9B when served by SGLang — and which one? The
earlier repo result ("MTP/n-gram slower, no speed lever, decode overhead-bound")
was measured only in ExLlamaV3, which lacks SGLang's GDN kernels and
ReplaySSM Ring Spec-Verify. This experiment re-measures the ceiling in the right
engine. It does not produce weights; it produces the decision for Exp 2.

## Fixed

Model, dtype, SGLang common flags, the four compared configs, the eval workload,
sampling (greedy), and the decode-tps definition are frozen in `config.py`. The
experiment does not tune flags per config to make one win beyond the small,
documented spec-decoding defaults it is testing.

## Configs compared (all same weights, same eval)

1. `baseline` — no speculation. Its greedy outputs are the losslessness reference.
2. `mtp` — Ornith's own native MTP head (distribution-matched to Ornith).
3. `dflash_stock` — DFlash draft trained on **stock** Qwen3.5-9B, not Ornith.
4. `eagle3_stock` — EAGLE3 draft trained on **stock** Qwen3.5-9B, not Ornith.

The tension is deliberate: (2) matches Ornith's distribution but is a weak
1-layer head; (3)/(4) are stronger drafts but trained on the pre-fine-tune
distribution. Which wins on Ornith is the open question.

## Metrics (verifiable, no prose similarity)

- **decode tok/s**, client-side, prefill excluded (streamed TTFT to last token).
- **acceptance** (mean accepted draft length) from SGLang `meta_info`.
- **GSM8K exact-match** and **HumanEval pass@1** (executed unit tests).
- **losslessness**: greedy spec output must be **token-identical** to the
  `baseline` greedy output on the same prompt. Report the exact-match fraction;
  a config that speeds up but changes outputs is reported as *not lossless*, not
  as a win.

## Honesty rules

- Absolute tok/s is host-dependent (the prior work saw 109 vs 182 on two 4090s).
  Lead with **within-run relative deltas** (config vs baseline on the same pod),
  not absolute numbers.
- Because greedy speculative decoding is exact, GSM8K/HumanEval scores should be
  identical across configs; that identity is a correctness check, not a
  per-config quality claim. The tps/acceptance differences are the result.
- If no spec config beats baseline in SGLang either, say so plainly — that is a
  strong, publishable negative result that closes the question, and it kills the
  Exp 2 draft-training path before it spends compute.

## Cost guard

One GPU pod, hard `$/hr` ceiling, wall-clock cap. The pod pushes the results
report to a private HF repo and self-terminates, so a lost local session cannot
leave it billing. No model weights are copied to the local machine.
