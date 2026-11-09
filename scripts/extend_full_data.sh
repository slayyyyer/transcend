#!/usr/bin/env bash
# Continue the full-Europarl run from its 350M-token checkpoint to 800M.
#
# The 350M run is not restarted. This wrapper waits for its training report,
# preserves the pre-extension artifacts, and resumes from checkpoint.pt.
# Token, step and walltime limits are cumulative except --max-walltime, which
# is a per-invocation budget.
set -euo pipefail

PY="${PYTHON:-/c/Users/justin/AppData/Local/Python/pythoncore-3.14-64/python.exe}"
CONFIG="${CONFIG:-configs/scaled_full.yaml}"
PREPARED="${PREPARED:-data/prepared/europarl_full}"
TOKENIZER="${TOKENIZER:-models/cpu_small/tokenizer}"
OUTPUT_DIR="${OUTPUT_DIR:-models/scaled_full}"
REPORT="${REPORT:-$OUTPUT_DIR/training_report.json}"
CHECKPOINT="${CHECKPOINT:-$OUTPUT_DIR/checkpoint.pt}"
LOG="${LOG:-reports/scaled_full_extend.log}"

MAX_TRAIN_TOKENS="${MAX_TRAIN_TOKENS:-800000000}"
MAX_STEPS="${MAX_STEPS:-100000}"
MAX_WALLTIME="${MAX_WALLTIME:-86400}"
PATIENCE="${PATIENCE:-30}"

WAIT="${WAIT:-1}"
DEADLINE_S="${DEADLINE_S:-21600}"
SETTLE_S="${SETTLE_S:-30}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$(dirname "$LOG")" "$OUTPUT_DIR"
log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"
}

test -s "$PREPARED/manifest.json"
test -s "$PREPARED/train.tsv"
test -s "$PREPARED/dev.tsv"
test -s "$TOKENIZER/merges.txt"

if [ "$WAIT" = "1" ] && [ ! -s "$REPORT" ]; then
  log "waiting for the current run to write $REPORT"
  started="$SECONDS"
  while [ ! -s "$REPORT" ]; do
    if (( SECONDS - started >= DEADLINE_S )); then
      log "timed out waiting for $REPORT after ${DEADLINE_S}s"
      exit 1
    fi
    sleep 30
  done
fi

test -s "$REPORT" || {
  log "missing training report: $REPORT (set WAIT=1 to wait for it)"
  exit 1
}
test -s "$CHECKPOINT" || {
  log "missing resume checkpoint: $CHECKPOINT"
  exit 1
}

# The current report is written after the last checkpoint. Give the training
# process a moment to exit before reading and copying the artifacts.
if [ "$WAIT" = "1" ]; then
  sleep "$SETTLE_S"
fi

STATE="$("$PY" - "$CHECKPOINT" <<'PY' | tr -d '\r'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(payload.get("global_step", 0)), int(payload.get("tokens_seen", 0)))
PY
)"
STEP="${STATE%% *}"
TOKENS="${STATE##* }"

if ! [[ "$STEP" =~ ^[0-9]+$ && "$TOKENS" =~ ^[0-9]+$ ]]; then
  log "could not read step/token state from $CHECKPOINT"
  exit 1
fi

log "resume state: step=$STEP tokens=$TOKENS"
if (( TOKENS >= MAX_TRAIN_TOKENS )); then
  log "checkpoint is already at or above the ${MAX_TRAIN_TOKENS}-token target"
  exit 0
fi
if (( STEP >= MAX_STEPS )); then
  log "checkpoint is already at step $STEP, at or above MAX_STEPS=$MAX_STEPS"
  exit 1
fi

if [ "$DRY_RUN" = "1" ]; then
  log "dry run: would resume $CHECKPOINT to tokens=$MAX_TRAIN_TOKENS steps=$MAX_STEPS walltime=${MAX_WALLTIME}s patience=$PATIENCE"
  exit 0
fi

# Keep the 350M artifacts available for the direct 350M-vs-800M comparison.
preserve() {
  local source="$1"
  local destination="$2"
  if [ -s "$source" ] && [ ! -e "$destination" ]; then
    cp -p "$source" "$destination"
    log "preserved $source -> $destination"
  fi
}
preserve "$REPORT" "$OUTPUT_DIR/training_report.pre_extend.json"
preserve "$OUTPUT_DIR/best.pt" "$OUTPUT_DIR/best.pre_extend.pt"
preserve "$CHECKPOINT" "$OUTPUT_DIR/checkpoint.pre_extend.pt"
preserve "$OUTPUT_DIR/history.json" "$OUTPUT_DIR/history.pre_extend.json"

log "resuming full-data training: target ${MAX_TRAIN_TOKENS} tokens / ${MAX_STEPS} updates / ${MAX_WALLTIME}s this invocation"
PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 "$PY" -m nmt.cli train \
  --config "$CONFIG" \
  --prepared "$PREPARED" \
  --tokenizer "$TOKENIZER" \
  --out "$OUTPUT_DIR" \
  --resume "$CHECKPOINT" \
  --device cuda \
  --max-train-tokens "$MAX_TRAIN_TOKENS" \
  --max-steps "$MAX_STEPS" \
  --max-walltime "$MAX_WALLTIME" \
  --patience "$PATIENCE" 2>&1 | tee -a "$LOG"

log "full-data extension finished"
