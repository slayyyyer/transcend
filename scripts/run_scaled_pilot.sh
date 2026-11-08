#!/usr/bin/env bash
# Run the controlled scaling experiment after the full final beam evaluation.
#
# The beam evaluation is already running under PID 3308 and writes its own log.
# This script waits for that log to finish, calibrates the larger model for 20
# updates, then starts the full run. Calibration catches 4 GiB VRAM failures
# before committing to a multi-hour job.
set -euo pipefail

PY="${PYTHON:-/c/Users/justin/AppData/Local/Python/pythoncore-3.14-64/python.exe}"
BEAM_LOG="${BEAM_LOG:-reports/newstest2014.82m.final_beam4.log}"
LOG="${LOG:-reports/scaled_pilot.log}"
CALIBRATION_DIR="${CALIBRATION_DIR:-models/scaled_pilot_calibration}"
OUTPUT_DIR="${OUTPUT_DIR:-models/scaled_pilot}"

mkdir -p reports
: > "$LOG"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"
}

wait_for_beam() {
  local deadline=$((SECONDS + 5400))
  while ! grep -q '^finished ' "$BEAM_LOG"; do
    if grep -q 'Traceback' "$BEAM_LOG"; then
      log "beam evaluation failed; not starting scaling"
      return 1
    fi
    if (( SECONDS >= deadline )); then
      log "timed out waiting for beam evaluation"
      return 1
    fi
    sleep 30
  done
}

run_train() {
  PYTHONIOENCODING=utf-8 "$PY" -m nmt.cli train \
    --config configs/scaled_accelerator.yaml \
    --prepared data/prepared/europarl_nc \
    --tokenizer models/cpu_small/tokenizer \
    --device cuda \
    "$@"
}

log "waiting for beam evaluation"
wait_for_beam
log "beam evaluation finished"

log "calibrating 19.8M model for 20 updates"
run_train \
  --out "$CALIBRATION_DIR" \
  --calibrate \
  --calibrate-updates 20 >> "$LOG" 2>&1

PYTHONIOENCODING=utf-8 "$PY" - "$CALIBRATION_DIR/calibration.json" <<'PY' >> "$LOG" 2>&1
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
    "peak_rss_bytes": report.get("peak_rss_bytes"),
}, indent=2))
PY

log "calibration passed; starting full scaled training"
run_train --out "$OUTPUT_DIR" >> "$LOG" 2>&1

log "scaled training finished"
