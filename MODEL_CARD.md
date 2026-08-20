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

# Ornith-1.5-9B with DFlash in SGLang

This repository contains no model weights. It documents a recipe for serving
[Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B) with the
[Qwen3.5-9B-DFlash](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash) draft model
in [SGLang](https://github.com/sgl-project/sglang).

## Results

Tested on one L40S with single-stream greedy decoding and 640-token
generations. GSM8K answers were checked numerically. HumanEval solutions were
run against their unit tests.

| configuration | decode tok/s | speedup | accepted tokens (GSM8K / HumanEval) | GSM8K | HumanEval |
|---|---:|---:|---:|---:|---:|
| Base model | 44.8 | 1.00x | — | 91.7% | 79.2% |
| DFlash, block size 16 | 207.5 | 4.63x | 6.5 / 8.5 | 90.8% | 80.8% |

Block size 16 was the fastest tested setting. The task pass rates stayed close
to the baseline in this sample. The benchmark does not claim identical text
across separate server runs.

## Run it

Use `lmsysorg/sglang:v0.5.17-cu129` with a GPU that has at least 24 GB of VRAM:

```bash
python -m sglang.launch_server \
  --model-path ornith-ai/Ornith-1.5-9B \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.5-9B-DFlash \
  --speculative-dflash-block-size 16 \
  --reasoning-parser qwen3 \
  --trust-remote-code
```

The tested setup used an L40S. An A100 also works. Keep `transformers` at
`5.12.1` with SGLang `0.5.17`.

Full reproduction files are in the
[GitHub repository](https://github.com/ultimatechris/ornith-dflash-sglang).

## Attribution

- Base model: [Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B)
- Draft model: [Qwen3.5-9B-DFlash](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash)
- Serving engine: [SGLang](https://github.com/sgl-project/sglang)

This recipe and benchmark were prepared by `ultimatechris`. The models keep
their own licenses.
