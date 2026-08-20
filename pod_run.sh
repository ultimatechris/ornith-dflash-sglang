#!/usr/bin/env bash
# Unattended pod pipeline for Exp 1: SGLang decode-tps clarifier on Ornith-1.5-9B.
#
# For each frozen config (baseline, mtp, dflash_stock, eagle3_stock) it launches
# an SGLang server, waits for readiness, runs the verifiable eval, stops the
# server, and moves on. Then it aggregates the report, pushes results to a
# private HF repo, and self-terminates. Survives a lost SSH session: stage log,
# wall-clock guard, per-config isolation, best-effort result upload on failure.
#
# Required env: HF_TOKEN. Optional: RUNPOD_API_KEY (+ RUNPOD_POD_ID) for teardown.
# Usage: bash pod_run.sh   (from the experiment dir on the pod)

set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
WORK="${WORK:-/workspace/ornith9b}"; mkdir -p "$WORK" "$HERE/results"
STATUS="$HERE/results/STATUS.txt"
PORT=30000
START=$(date +%s); MAX_SECONDS="${MAX_SECONDS:-10800}"   # 3h guard

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
stage(){ echo "$1" > "$STATUS"; log "STAGE: $1"; }
elapsed(){ echo $(( $(date +%s) - START )); }
guard(){ [ "$(elapsed)" -gt "$MAX_SECONDS" ] && { log "WALL-CLOCK GUARD $(elapsed)s"; return 1; }; return 0; }
cfg(){ python3 -c "import config; print(config.$1)"; }

upload_results(){
  [ -n "${HF_TOKEN:-}" ] || return 0
  python3 - <<'PY' 2>/dev/null || true
import os, glob, config
from huggingface_hub import HfApi, create_repo
tok=os.environ.get("HF_TOKEN")
if not tok: raise SystemExit(0)
api=HfApi(token=tok); repo=config.HF_RESULTS_REPO
create_repo(repo, private=config.HF_PRIVATE, repo_type="model", exist_ok=True, token=tok)
for f in (glob.glob("results/*.json")+glob.glob("results/*.md")+glob.glob("results/*.txt")
          +glob.glob("results/*.log")+["config.py","README.md"]):
    if os.path.exists(f):
        try: api.upload_file(path_or_fileobj=f, path_in_repo=("logs/"+os.path.basename(f) if f.endswith(".log") else os.path.basename(f)), repo_id=repo, token=tok)
        except Exception as e: print("skip",f,e)
PY
}

self_terminate(){
  if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    log "Self-terminating pod ${RUNPOD_POD_ID}"
    curl -s "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' \
      -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}" >/dev/null 2>&1 || true
  else log "No RUNPOD creds/pod id; leaving pod up."; fi
}
fail(){ stage "FAILED:$1"; log "FATAL: $1"; upload_results || true; log "Pod left UP for inspection."; exit 1; }

SERVER_PID=""
stop_server(){ [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null; pkill -f sglang.launch_server 2>/dev/null; sleep 10; SERVER_PID=""; }
launch_server(){  # $1=name ; rest=extra flags
  local name="$1"; shift
  python -m sglang.launch_server --model-path "$MODELDIR" --host 127.0.0.1 --port "$PORT" \
      --trust-remote-code "${COMMON[@]}" "$@" > "results/server_${name}.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 180); do   # up to ~15 min for first cold start
    curl -sf "http://127.0.0.1:${PORT}/get_model_info" >/dev/null 2>&1 && { log "server ready ($name)"; return 0; }
    kill -0 "$SERVER_PID" 2>/dev/null || { log "server died ($name)"; return 1; }
    sleep 5
  done
  log "server readiness timeout ($name)"; return 1
}

# --------------------------------------------------------------------------
stage "deps"
# Run on the official SGLang image (lmsysorg/sglang:v0.5.17-cu129), which ships
# correctly-built kernels for all archs incl. sm80 (A100). If sglang already
# imports, do NOT reinstall it — an --upgrade would clobber the matched binaries.
# Only the small eval extras are added. transformers must stay at 5.12.1 (matches
# sglang 0.5.17 and the Ornith/Qwen3.5 config; 5.15 collides on qwen3_asr).
export DEBIAN_FRONTEND=noninteractive HF_HUB_ENABLE_HF_TRANSFER=1
export PIP_BREAK_SYSTEM_PACKAGES=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1   # recommended by the DFlash model card
if python3 -c "import sglang, sgl_kernel" 2>/dev/null; then
  log "sglang+sgl_kernel already present; skipping sglang install"
else
  pip install --upgrade pip || true
  pip install --upgrade "sglang[all]" || fail "pip sglang"
fi
pip install --upgrade datasets requests "huggingface_hub[cli]" hf_transfer || fail "pip eval deps"
python3 -c "import transformers,sys; sys.exit(0 if transformers.__version__.startswith('5.12') else 1)" \
  || pip install "transformers==5.12.1" || fail "pin transformers"
python3 -c "import sglang, transformers, sgl_kernel; print('sglang', sglang.__version__, 'transformers', transformers.__version__, 'sgl_kernel OK')" || fail "import check"

stage "download-model"
MODELDIR="$WORK/base_model"
TARGET="$(cfg TARGET_MODEL)"
[ -f "$MODELDIR/config.json" ] || hf download "$TARGET" --local-dir "$MODELDIR" >/dev/null 2>&1 || fail "download $TARGET"
log "model at $MODELDIR"

mapfile -t COMMON < <(python3 -c "import config; print('\n'.join(config.SGLANG_COMMON))")

# --------------------------------------------------------------------------
stage "sweep"
mapfile -t NAMES < <(python3 -c "import config; print('\n'.join(config.SPEC_CONFIGS))")
for name in "${NAMES[@]}"; do
  guard || { log "guard: stop before $name"; break; }
  [ -f "results/${name}.json" ] && { log "skip done $name"; continue; }
  # per-element print: an empty spec list yields zero lines (not one empty line),
  # so SPEC becomes an empty array instead of a single "" arg that sglang rejects.
  mapfile -t SPEC < <(python3 -c "import config,sys; [print(x) for x in config.SPEC_CONFIGS[sys.argv[1]]]" "$name")
  log "=== config: $name (flags: ${SPEC[*]:-none}) ==="
  if launch_server "$name" "${SPEC[@]}"; then
    python3 eval_harness.py run --config "$name" --url "http://127.0.0.1:${PORT}" --model-dir "$MODELDIR" \
      > "results/eval_${name}.log" 2>&1 || log "eval FAILED for $name (continuing)"
  else
    log "launch FAILED for $name (continuing); see results/server_${name}.log"
  fi
  stop_server
  upload_results || true   # push partial after each config
done

# --------------------------------------------------------------------------
stage "report"
python3 eval_harness.py report > results/report.out 2>&1 || log "report parse issue"
cat results/report.out || true

stage "upload"
upload_results || fail "upload"

stage "DONE"
log "SUCCESS. Report on HF: $(cfg HF_RESULTS_REPO). Elapsed $(elapsed)s."
self_terminate
