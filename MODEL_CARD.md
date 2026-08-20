---
base_model:
  - ornith-ai/Ornith-1.5-9B
  - z-lab/Qwen3.5-9B-DFlash
library_name: sglang
tags:
  - speculative-decoding
  - dflash
  - sglang
  - qwen3.5
  - inference-optimization
  - benchmark
license: mit
pipeline_tag: text-generation
---

# Ornith-1.5-9B + DFlash: ~4.6x faster decoding (SGLang recipe)

This repo has no new weights. It's a recipe and a benchmark. Serving
[`ornith-ai/Ornith-1.5-9B`](https://huggingface.co/ornith-ai/Ornith-1.5-9B) in
SGLang with the [`z-lab/Qwen3.5-9B-DFlash`](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash)
draft makes single-stream decoding about 4.6x faster, losslessly, without
touching the model. Nothing official from Ornith AI, z-lab, or SGLang; it's
their pieces put together and measured.

## Numbers

One L40S, single stream, greedy, 640-token generations. GSM8K (n=120) and
HumanEval (unit tests executed, n=120):

| config | decode tok/s | speedup | accept len (math / code) | GSM8K | HumanEval |
|---|---:|---:|---:|---:|---:|
| baseline | 44.8 | 1.00x | n/a | 91.7% | 79.2% |
| DFlash, block 16 | 207.5 | 4.63x | 6.5 / 8.5 | 90.8% | 80.8% |

Code gains more than math: about 5.5x on HumanEval versus 4.2x on GSM8K. tok/s
depends on your GPU, so the speedup is the number that transfers. Pass rates
hold, so quality is intact. The model's own MTP head, by contrast, accepts
almost nothing and gives no speedup at all. The trained draft is what does the
work.

## How to run it

```bash
python -m sglang.launch_server \
  --model-path ornith-ai/Ornith-1.5-9B \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.5-9B-DFlash \
  --speculative-dflash-block-size 16 \
  --reasoning-parser qwen3 --trust-remote-code
```

Run it inside `lmsysorg/sglang:v0.5.17-cu129`. That image has the sm80+ kernels,
whereas the default `pip install sglang[all]` ships CUDA-13/Blackwell-only ones.
You need a GPU with 24 GB or more, and pin `transformers==5.12.1`.

## Method, harness, and the gotchas

The full writeup, the benchmark code, and the config sweep that ruled everything
else out (native MTP, and the DFlash-as-EAGLE misconfiguration that just crashes)
are in the GitHub repo: https://github.com/ultimatechris/ornith-dflash-sglang

## Credit

- Base model: Ornith AI, `ornith-ai/Ornith-1.5-9B`
- DFlash draft: z-lab, `z-lab/Qwen3.5-9B-DFlash`
- Engine: the SGLang project

Benchmark by `ultimatechris`. The models keep their own licenses.
