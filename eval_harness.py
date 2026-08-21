"""Verifiable eval + decode-tps measurement against a live SGLang server.

Modes:
    run    --config <name> --url <sglang_url> --model-dir <path>
           Runs the frozen workload against the server for one config, writes
           results/<config>.json (+ results/baseline_outputs.json for baseline).
    report Aggregates every results/<config>.json, computes losslessness vs the
           baseline outputs, writes results/report.json and results/REPORT.md.

Metrics are verifiable: GSM8K exact numeric match, HumanEval executed pass@1,
decode tok/s from streamed timing (prefill excluded), acceptance from SGLang
meta_info, and greedy token-identity vs baseline as the losslessness check.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

import importlib
config = importlib.import_module(os.environ.get("ORNITH_CONFIG", "config_9b"))

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Prompts / workload
# ---------------------------------------------------------------------------
def load_tokenizer(model_dir: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


def chat(tok, user: str) -> str:
    msgs = [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def gsm8k_items(n: int):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for ex in ds.select(range(min(n, len(ds)))):
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        q = (ex["question"].strip()
             + "\n\nSolve step by step. End your reply with a line "
               "'#### <final integer answer>'.")
        items.append({"task": "gsm8k", "id": ex["question"][:40], "prompt": q, "gold": gold})
    return items


def humaneval_items(n: int):
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    items = []
    for ex in ds.select(range(min(n, len(ds)))):
        q = ("Complete this Python function. Return ONLY the complete function "
             "in a single ```python code block, no explanation.\n\n"
             + ex["prompt"])
        items.append({
            "task": "humaneval", "id": ex["task_id"], "prompt": q,
            "he_prompt": ex["prompt"], "test": ex["test"], "entry_point": ex["entry_point"],
        })
    return items


# ---------------------------------------------------------------------------
# SGLang generation with streamed decode-tps + acceptance
# ---------------------------------------------------------------------------
def generate(url: str, prompt: str, max_new_tokens: int) -> dict:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0, "max_new_tokens": max_new_tokens, "repetition_penalty": 1.0,
        },
        "stream": True,
    }
    t0 = time.perf_counter()
    t_first = None
    t_last = None
    text = ""
    meta = {}
    with requests.post(url.rstrip("/") + "/generate", json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            now = time.perf_counter()
            new_text = chunk.get("text", "")
            if new_text and t_first is None:
                t_first = now
            if new_text:
                t_last = now
                text = new_text  # SGLang streams cumulative text
            if "meta_info" in chunk and chunk["meta_info"]:
                meta = chunk["meta_info"]
    completion_tokens = int(meta.get("completion_tokens") or 0)
    # decode tps excludes prefill: first token -> last token window
    decode_tps = None
    if t_first is not None and t_last is not None and t_last > t_first and completion_tokens > 1:
        decode_tps = (completion_tokens - 1) / (t_last - t_first)
    # acceptance (spec configs only)
    accept_len = meta.get("accept_length")
    if accept_len is None:
        sv = meta.get("spec_verify_ct")
        if sv and completion_tokens:
            try:
                accept_len = completion_tokens / float(sv)
            except ZeroDivisionError:
                accept_len = None
    return {
        "text": text,
        "completion_tokens": completion_tokens,
        "ttft_s": (t_first - t0) if t_first else None,
        "decode_tps": decode_tps,
        "accept_length": accept_len,
        "e2e_s": (t_last - t0) if t_last else None,
    }


# ---------------------------------------------------------------------------
# Verifiable scoring
# ---------------------------------------------------------------------------
def score_gsm8k(text: str, gold: str) -> bool:
    m = re.search(r"####\s*(-?\d[\d,]*)", text)
    if m:
        pred = m.group(1).replace(",", "")
    else:
        nums = re.findall(r"-?\d[\d,]*", text)
        if not nums:
            return False
        pred = nums[-1].replace(",", "")
    try:
        return int(pred) == int(gold)
    except ValueError:
        return pred == gold


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (2_000_000_000, 2_000_000_000))
    except (ValueError, OSError):
        pass


def score_humaneval(text: str, item: dict) -> bool:
    code = extract_code(text)
    programs = [code, item["he_prompt"] + "\n" + code]  # try full fn, then completion
    check = item["test"] + f"\ncheck({item['entry_point']})\n"
    for body in programs:
        if item["entry_point"] not in body:
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(body + "\n" + check)
            path = f.name
        try:
            p = subprocess.run([sys.executable, path], capture_output=True,
                               timeout=15, preexec_fn=_limits)
            if p.returncode == 0:
                os.unlink(path)
                return True
        except (subprocess.TimeoutExpired, Exception):
            pass
        finally:
            if os.path.exists(path):
                os.unlink(path)
    return False


# ---------------------------------------------------------------------------
# Run one config
# ---------------------------------------------------------------------------
def run_config(name: str, url: str, model_dir: str):
    tok = load_tokenizer(model_dir)
    items = gsm8k_items(config.GSM8K_N) + humaneval_items(config.HUMANEVAL_N)
    out_path = RESULTS / f"{name}.json"
    records = []
    for i, it in enumerate(items):
        prompt = chat(tok, it["prompt"])
        try:
            g = generate(url, prompt, config.MAX_NEW_TOKENS)
        except Exception as e:  # keep going; a single failure must not abort
            g = {"text": "", "completion_tokens": 0, "decode_tps": None,
                 "accept_length": None, "error": str(e)}
        ok = None
        if it["task"] == "gsm8k":
            ok = score_gsm8k(g["text"], it["gold"])
        else:
            ok = score_humaneval(g["text"], it)
        records.append({
            "task": it["task"], "id": it["id"], "correct": bool(ok),
            "decode_tps": g["decode_tps"], "accept_length": g["accept_length"],
            "completion_tokens": g["completion_tokens"], "ttft_s": g.get("ttft_s"),
            "text": g["text"],
        })
        if i % 10 == 0:
            out_path.write_text(json.dumps({"config": name, "records": records}, indent=2))
            print(f"[{name}] {i+1}/{len(items)}  last tps={g['decode_tps']}")
    out_path.write_text(json.dumps({"config": name, "records": records}, indent=2))
    # baseline outputs are the losslessness reference
    if name == config.BASELINE:
        ref = {r["task"] + "|" + str(r["id"]): r["text"] for r in records}
        (RESULTS / "baseline_outputs.json").write_text(json.dumps(ref, indent=2))
    print(f"[{name}] done, wrote {out_path}")


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------
def _agg(records, task):
    rs = [r for r in records if r["task"] == task]
    tps = [r["decode_tps"] for r in rs if r["decode_tps"]]
    acc = [r["accept_length"] for r in rs if r["accept_length"]]
    correct = [r for r in rs if r["correct"]]
    return {
        "n": len(rs),
        "median_decode_tps": round(statistics.median(tps), 2) if tps else None,
        "mean_accept_length": round(statistics.mean(acc), 3) if acc else None,
        "pass_rate": round(len(correct) / len(rs), 4) if rs else None,
    }


def report():
    ref = {}
    ref_path = RESULTS / "baseline_outputs.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
    base_tps = {}
    summary = {}
    for name in config.SPEC_CONFIGS:
        p = RESULTS / f"{name}.json"
        if not p.exists():
            continue
        recs = json.loads(p.read_text())["records"]
        entry = {"gsm8k": _agg(recs, "gsm8k"), "humaneval": _agg(recs, "humaneval")}
        # losslessness vs baseline outputs (greedy => should be identical)
        if name != config.BASELINE and ref:
            same = tot = 0
            for r in recs:
                k = r["task"] + "|" + str(r["id"])
                if k in ref:
                    tot += 1
                    same += int(r["text"] == ref[k])
            entry["lossless_exact_frac"] = round(same / tot, 4) if tot else None
        all_tps = [r["decode_tps"] for r in recs if r["decode_tps"]]
        entry["overall_median_decode_tps"] = round(statistics.median(all_tps), 2) if all_tps else None
        summary[name] = entry
    # relative speedups vs baseline
    base = summary.get(config.BASELINE, {}).get("overall_median_decode_tps")
    for name, e in summary.items():
        if base and e.get("overall_median_decode_tps"):
            e["speedup_vs_baseline"] = round(e["overall_median_decode_tps"] / base, 3)
    (RESULTS / "report.json").write_text(json.dumps(summary, indent=2))
    (RESULTS / "REPORT.md").write_text(render_report(summary, base))
    print(json.dumps(summary, indent=2))


def render_report(summary, base_tps):
    def g(e, k, d="—"):
        v = e.get(k)
        return d if v is None else v

    L = ["# Exp 1 — Ornith-1.5-9B decode tok/s in SGLang\n",
         f"Target: `{config.TARGET_MODEL}` ({config.DTYPE}), single stream, greedy. "
         f"GSM8K n={config.GSM8K_N} (exact match), HumanEval n={config.HUMANEVAL_N} "
         "(executed pass@1). Decode tps excludes prefill; acceptance from SGLang "
         "meta_info; losslessness = greedy token-identity vs the baseline.\n",
         "## Result\n",
         "| config | median decode tps | speedup | mean accept len | lossless | GSM8K | HumanEval |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for name in config.SPEC_CONFIGS:
        e = summary.get(name)
        if not e:
            continue
        acc = (e.get("humaneval", {}).get("mean_accept_length")
               or e.get("gsm8k", {}).get("mean_accept_length"))
        L.append(
            f"| `{name}` | {g(e,'overall_median_decode_tps')} | "
            f"{g(e,'speedup_vs_baseline')} | {acc if acc else '—'} | "
            f"{g(e,'lossless_exact_frac')} | "
            f"{e.get('gsm8k',{}).get('pass_rate')} | {e.get('humaneval',{}).get('pass_rate')} |"
        )
    L += ["",
          "## Reading this\n",
          "- Speedup is the within-run median decode-tps ratio vs `baseline` on "
          "the same pod (absolute tps is host-dependent, so trust the ratio).\n",
          "- GSM8K/HumanEval pass rates should be ~identical across configs "
          "(greedy + lossless); differences there mean a config was **not** "
          "lossless and its speedup does not count.\n",
          "- `mtp` = Ornith's own head (distribution-matched); `dflash_stock` / "
          "`eagle3_stock` = drafts trained on base Qwen3.5. If the stock drafts "
          "lose acceptance to fine-tune drift, that motivates Exp 2 (an "
          "Ornith-specific draft).\n"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "report"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--model-dir", default=None)
    a = ap.parse_args()
    if a.mode == "run":
        run_config(a.config, a.url, a.model_dir)
    else:
        report()


if __name__ == "__main__":
    main()
