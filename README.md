# Ornith-1.5-9B with DFlash in SGLang

This repository contains the commands, evaluation harness, and results for
running [Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B) with the
[Qwen3.5-9B-DFlash](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash) draft model
in [SGLang](https://github.com/sgl-project/sglang).

There are no new model weights here. The recipe combines the base model, the
DFlash draft, and SGLang's speculative-decoding support.

## Results

The test used one L40S, one request at a time, greedy decoding, and 640-token
generations. GSM8K answers were checked numerically. HumanEval solutions were
run against their unit tests.

| configuration | decode tok/s | speedup | accepted tokens (GSM8K / HumanEval) | GSM8K | HumanEval |
|---|---:|---:|---:|---:|---:|
| Base model | 44.8 | 1.00x | — | 91.7% | 79.2% |
| DFlash, block size 16 | 207.5 | 4.63x | 6.5 / 8.5 | 90.8% | 80.8% |

Block size 16 was the fastest of the tested settings. Block sizes 24 and 32
accepted slightly more tokens but produced lower overall throughput. The
absolute tok/s figure depends on the GPU; the within-run speedup is the useful
comparison.

The task pass rates stayed close to the baseline in this sample. This is a
benchmark result, not a claim that separate server runs will produce identical
text for every prompt.

## Run the model

Use the SGLang image with the matching CUDA kernels:

```bash
docker run --gpus all -it lmsysorg/sglang:v0.5.17-cu129
```

Inside the container, start SGLang with:

```bash
python -m sglang.launch_server \
  --model-path ornith-ai/Ornith-1.5-9B \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.5-9B-DFlash \
  --speculative-dflash-block-size 16 \
  --reasoning-parser qwen3 \
  --trust-remote-code
```

The recipe needs a GPU with at least 24 GB of VRAM. The tested setup used an
L40S. An A100 also works.

The SGLang image is recommended because it includes kernels for Ampere and
newer GPUs. Keep `transformers` at `5.12.1` with SGLang `0.5.17`.

## Reproduce the evaluation

Start SGLang with one configuration at a time, then run `eval_harness.py`.
For each running server, the harness measures:

- decode throughput, excluding prefill
- accepted draft length from SGLang
- GSM8K exact-match accuracy
- HumanEval pass@1 using the provided unit tests

The server flags and benchmark sizes are in `config.py`. For example:

```bash
python eval_harness.py run --config baseline \
  --url http://127.0.0.1:30000 --model-dir /path/to/base_model
python eval_harness.py run --config dflash_bs16 \
  --url http://127.0.0.1:30000 --model-dir /path/to/base_model
python eval_harness.py report
```

The harness generates the result files locally.

## License

The benchmark code is MIT-licensed. The base model and DFlash draft keep their
own licenses; read their model cards before using them.
