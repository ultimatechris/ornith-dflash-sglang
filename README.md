# Ornith-1.5 with DFlash in SGLang

This repository contains the commands, evaluation harness, and results for
running the [Ornith-1.5](https://huggingface.co/ornith-ai) models with their
matching [DFlash](https://huggingface.co/z-lab) draft models in
[SGLang](https://github.com/sgl-project/sglang).

There are no new model weights here. Each recipe combines a base model, a DFlash
draft, and SGLang's speculative-decoding support.

## Results

The tests used one request at a time, greedy decoding, and 640-token
generations. GSM8K answers were checked numerically. HumanEval solutions were
run against their unit tests. The absolute tok/s depends on the GPU; the
within-run speedup is the useful comparison. The task pass rates stayed close to
the baseline in both cases.

### Ornith-1.5-9B

One L40S.

| configuration | decode tok/s | speedup | accepted tokens (GSM8K / HumanEval) | GSM8K | HumanEval |
|---|---:|---:|---:|---:|---:|
| Base model | 44.8 | 1.00x | — | 91.7% | 79.2% |
| DFlash, block size 16 | 207.5 | 4.63x | 6.5 / 8.5 | 90.8% | 80.8% |

### Ornith-1.5-35B-A3B

Two A100 80GB.

| configuration | decode tok/s | speedup | accepted tokens (GSM8K / HumanEval) | GSM8K | HumanEval |
|---|---:|---:|---:|---:|---:|
| Base model | 202.5 | 1.00x | — | 93.3% | 87.5% |
| DFlash, block size 16 | 470.2 | 2.32x | 6.1 / 7.7 | 90.0% | 90.0% |

The 35B is a sparse MoE with about 3B active parameters, so its baseline is
already fast and the speedup is smaller than the dense 9B's. Block size 16 was
the fastest tested setting for both models.

## Run the model

Use the SGLang image with the matching CUDA kernels, and keep `transformers` at
`5.12.1` with SGLang `0.5.17`:

```bash
docker run --gpus all -it lmsysorg/sglang:v0.5.17-cu129
```

### Ornith-1.5-9B

Needs one GPU with at least 24 GB of VRAM. The tested setup used an L40S; an
A100 also works.

```bash
python -m sglang.launch_server \
  --model-path ornith-ai/Ornith-1.5-9B \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.5-9B-DFlash \
  --speculative-dflash-block-size 16 \
  --reasoning-parser qwen3 \
  --trust-remote-code
```

### Ornith-1.5-35B-A3B

The bf16 weights are about 73 GB. On two 80 GB GPUs:

```bash
python -m sglang.launch_server \
  --model-path ornith-ai/Ornith-1.5-35B-A3B \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.5-35B-A3B-DFlash \
  --speculative-dflash-block-size 16 \
  --reasoning-parser qwen3 \
  --tp 2 \
  --mem-fraction-static 0.80 \
  --max-total-tokens 40960 \
  --max-running-requests 4 \
  --mm-feature-transport cpu \
  --trust-remote-code
```

On a single GPU with about 90 GB or more, use `--tp 1` instead. A single 80 GB
card is not enough for the bf16 weights. The last four flags keep the KV pool and
state cache small enough for the CUDA graphs, and avoid a syscall some hosts
block for the vision feature transport (this recipe sends only text).

## Reproduce the evaluation

Start SGLang with one configuration at a time, then run `eval_harness.py`.
For each running server it measures:

- decode throughput, excluding prefill
- accepted draft length from SGLang
- GSM8K exact-match accuracy
- HumanEval pass@1 using the provided unit tests

The server flags and benchmark sizes are in `config_9b.py` and `config_35b.py`.
Pick one with the `ORNITH_CONFIG` variable (it defaults to `config_9b`):

```bash
ORNITH_CONFIG=config_9b python eval_harness.py run --config baseline \
  --url http://127.0.0.1:30000 --model-dir /path/to/base_model
ORNITH_CONFIG=config_9b python eval_harness.py run --config dflash_bs16 \
  --url http://127.0.0.1:30000 --model-dir /path/to/base_model
ORNITH_CONFIG=config_9b python eval_harness.py report
```

Use `ORNITH_CONFIG=config_35b` for the 35B. The harness writes the result files
locally.

## License

The benchmark code is MIT-licensed. The base models and DFlash drafts keep their
own licenses; read their model cards before using them.
