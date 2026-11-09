#!/usr/bin/env bash
# Start the controlled full-Europarl data-expansion run.
#
# Prerequisites: data/prepared/europarl_full was built with --europarl-stride 1
# and --news-commentary-stride 1. This script calibrates the existing 19.8M
# architecture on the new dataset, then trains from scratch for at most 350M
# target tokens, 45k updates, or 12 hours for this invocation.
set -euo pipefail

PY="${PYTHON:-/c/Users/justin/AppData/Local/Python/pythoncore-3.14-64/python.exe}"
CONFIG="${CONFIG:-configs/scaled_full.yaml}"
PREPARED="${PREPARED:-data/prepared/europarl_full}"
TOKENIZER="${TOKENIZER:-models/cpu_small/tokenizer}"
OUTPUT_DIR="${OUTPUT_DIR:-models/scaled_full}"
CALIBRATION_DIR="${CALIBRATION_DIR:-models/scaled_full_calibration}"
LOG="${LOG:-reports/scaled_full.log}"

mkdir -p reports
: > "$LOG"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"
}

run_train() {
  PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 "$PY" -m nmt.cli train \
    --config "$CONFIG" \
    --prepared "$PREPARED" \
    --tokenizer "$TOKENIZER" \
    --device cuda \
    "$@" 2>&1 | tee -a "$LOG"
}

test -s "$PREPARED/manifest.json"
test -s "$PREPARED/train.tsv"
test -s "$PREPARED/dev.tsv"
test -s "$TOKENIZER/merges.txt"

if [ -s "$OUTPUT_DIR/checkpoint.pt" ]; then
  log "refusing to overwrite existing checkpoint in $OUTPUT_DIR"
  exit 1
fi

log "full-data manifest: $PREPARED/manifest.json"
"$PY" - "$PREPARED/manifest.json" <<'PY' | tee -a "$LOG"
import json
import sys

manifest = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(json.dumps({
    "sampling": manifest.get("sampling"),
    "counts": manifest.get("counts"),
    "cleaning": manifest.get("cleaning"),
}, indent=2, sort_keys=True))
PY

log "calibrating on full prepared split"
run_train --out "$CALIBRATION_DIR" --calibrate --calibrate-updates 20

PYTHONIOENCODING=utf-8 "$PY" - "$CALIBRATION_DIR/calibration.json" <<'PY' | tee -a "$LOG"
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("stop_reason") != "calibration_complete":
    raise SystemExit(f"calibration did not complete: {report.get('stop_reason')}")
print(json.dumps({
    "parameter_count": report.get("parameter_count"),
    "mean_update_ms": report.get("mean_update_ms"),
    "target_tokens_per_second": report.get("target_tokens_per_second"),
    "tokens_this_run": report.get("tokens_this_run"),
}, indent=2, sort_keys=True))
PY

log "starting full-data training: 350M tokens / 45k updates / 12h"
run_train \
  --out "$OUTPUT_DIR" \
  --max-train-tokens 350000000 \
  --max-steps 45000 \
  --max-walltime 43200

log "full-data training finished"
