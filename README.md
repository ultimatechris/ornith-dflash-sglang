# Ornith-1.5-9B: ~4.6x faster decoding with DFlash in SGLang

Run [`ornith-ai/Ornith-1.5-9B`](https://huggingface.co/ornith-ai/Ornith-1.5-9B)
in SGLang with the [`z-lab/Qwen3.5-9B-DFlash`](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash)
draft and single-stream decoding gets about 4.6x faster, with no measurable
quality loss and no changes to the model. The rest of this repo is how we got
there and how you reproduce it.

It isn't an official release from anyone. We paired existing pieces: the base
model is Ornith AI's, the DFlash draft is z-lab's, the engine is SGLang.

## The numbers

One L40S, single stream, greedy, 640-token generations. GSM8K (exact-match,
n=120) and HumanEval (unit tests executed, n=120):

| config | decode tok/s | speedup | accept len (math / code) | GSM8K | HumanEval |
|---|---:|---:|---:|---:|---:|
| baseline | 44.8 | 1.00x | n/a | 91.7% | 79.2% |
| DFlash, block 16 | 207.5 | 4.63x | 6.5 / 8.5 | 90.8% | 80.8% |

Code speeds up more than math: HumanEval hits about 5.5x, GSM8K about 4.2x. The
draft guesses predictable code better than chained arithmetic. Block size 16 was
the fastest; 24 and 32 accept a few more tokens but end up slower overall. The
pass rates barely move, so quality holds. tok/s depends on your GPU, so the
speedup is the number that carries over, not the absolute rate.

## Why it works, and why the obvious thing doesn't

Ornith-1.5-9B is a Qwen3.5 model: gated-DeltaNet linear attention, a sparse MoE,
and a built-in 1-layer MTP (multi-token-prediction) head. The obvious thing to
try is turning that MTP head on for speculative decoding. It doesn't help. We
swept every NEXTN setting we could (draft length, tree width, ReplaySSM on and
off) and draft acceptance never rose above about 1.0. Nothing gets accepted, so
you pay for the draft and get nothing back, and every config came out slower
than baseline. Older ExLlamaV3 experiments reached the same verdict, so it isn't
an engine quirk.

The fix wasn't a better engine or a cleverer flag; it was a better draft. DFlash
is trained for this job and accepts 6 to 9 tokens a step, which is where the
4.6x comes from. This DFlash draft was trained on stock Qwen3.5, not on Ornith's
fine-tune, and it still works well. A draft trained on Ornith directly would
probably do better.

## Reproduce it

You need one GPU with 24 GB or more (an L40S or A100 is plenty; the DFlash
kernels want Ampere or newer, that is sm80+). Grab the official SGLang CUDA-12
image so the kernels match your card:

```bash
docker run --gpus all -it lmsysorg/sglang:v0.5.17-cu129
```

Then start the server with the draft:

```bash
python -m sglang.launch_server \
  --model-path ornith-ai/Ornith-1.5-9B \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.5-9B-DFlash \
  --speculative-dflash-block-size 16 \
  --reasoning-parser qwen3 \
  --trust-remote-code
```

To reproduce the whole table, `pod_run.sh` runs the full sweep on a fresh GPU pod
and `eval_harness.py` does the measuring: decode tok/s (streamed, prefill
excluded), acceptance from SGLang's `meta_info`, GSM8K exact-match, HumanEval
executed pass@1. Compare against a plain `--model-path ornith-ai/Ornith-1.5-9B`
server for your baseline.

## Gotchas

- DFlash is `--speculative-algorithm DFLASH`, not `EAGLE`. Point EAGLE at a
  DFlash draft and it dies with `'DFlashDraftModel' object has no attribute
  set_embed_and_head`. DFlash also uses `--speculative-dflash-block-size`, not
  `--speculative-num-steps` / `--speculative-num-draft-tokens`.
- On an A100 the plain `pip install sglang[all]` won't run this. Its default is
  CUDA 13 with Blackwell-only (sm100) kernels and no sm80 build; see sglang#11421.
  The `lmsysorg/sglang:v0.5.17-cu129` image has the sm80 kernels, so use it.
- Pin `transformers==5.12.1`. It matches sglang 0.5.17 and the Ornith/Qwen3.5
  config. 5.15 crashes at import over the `qwen3_asr` registration.
- The flashinfer linear-attn backends need sm90+. Leave them off and plain
  DFlash runs fine on Ampere and Ada.
- Block size has a sweet spot. 16 was fastest for us; 8 accepts less, 24 and 32
  accept a bit more but net out slower. Start at 16.

## What's in here

- `pod_run.sh` sets up the box, pulls the model, sweeps the configs, benchmarks
  each, and writes a leaderboard.
- `eval_harness.py` does the measuring: decode tok/s, acceptance, GSM8K, HumanEval.
- `config.py` holds the frozen bits: model, flags, config grid, eval sizes.
- `program.md` is the rules the sweep followed.
- `results.json` is the confirmation run behind the table above.

## License

MIT for this benchmark code. The models keep their own licenses, so check their
pages.
