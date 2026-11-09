#!/usr/bin/env bash
# Evaluate the full-data checkpoint against scaled step 18,000.
#
# The full run writes training_report.json only after it stops. Waiting for that
# report keeps evaluation from competing with training for the GPU. If you are
# rerunning an already-finished experiment, WAIT=0 skips the wait.
set -euo pipefail

PY="${PYTHON:-/c/Users/justin/AppData/Local/Python/pythoncore-3.14-64/python.exe}"
TRAIN_REPORT="${TRAIN_REPORT:-models/scaled_full/training_report.json}"
CANDIDATE="${CANDIDATE:-models/scaled_full/best.pt}"
BASELINE="${BASELINE:-models/scaled_pilot/best.pt}"
CONFIG="${CONFIG:-configs/scaled_full.yaml}"
TOKENIZER="${TOKENIZER:-models/cpu_small/tokenizer}"
BENCHMARK_DIR="${BENCHMARK_DIR:-data/prepared/newstest2014}"
REPORTS_DIR="${REPORTS_DIR:-reports}"
LOG="${LOG:-reports/scaled_full_eval.log}"
TAG="${TAG:-scaled_full}"
DEADLINE_S="${DEADLINE_S:-43200}"
SETTLE_S="${SETTLE_S:-60}"
DEVICE="${DEVICE:-cuda}"
LIMIT="${LIMIT:-0}"
BOOTSTRAP="${BOOTSTRAP:-1000}"
WAIT="${WAIT:-1}"
OUT="${OUT:-reports/newstest2014.${TAG}.comparisons.json}"
MARKDOWN="${MARKDOWN:-reports/newstest2014.${TAG}.comparisons.md}"

mkdir -p "$REPORTS_DIR"
: > "$LOG"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"
}

wait_for_training() {
  if [ "$WAIT" != "1" ]; then
    log "WAIT=$WAIT; not waiting for the training report"
    return 0
  fi
  local deadline=$((SECONDS + DEADLINE_S))
  while [ ! -f "$TRAIN_REPORT" ]; do
    if (( SECONDS >= deadline )); then
      log "timed out waiting for $TRAIN_REPORT"
      return 1
    fi
    sleep 30
  done
  # training_report.json is written just before the training process exits.
  sleep "$SETTLE_S"
}

run_eval() {
  local name="$1"
  local direction="$2"
  shift 2
  local out="$REPORTS_DIR/newstest2014.${TAG}.${name}.${direction}.full.json"
  local -a limit_args=()
  if [ "$LIMIT" != "0" ]; then
    limit_args=(--limit "$LIMIT")
  fi
  log "starting ${name} ${direction} -> ${out}"
  PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 "$PY" -m nmt.cli evaluate \
    --config "$CONFIG" \
    --tokenizer "$TOKENIZER" \
    --benchmark "$BENCHMARK_DIR/newstest2014.${direction}.jsonl" \
    --direction "$direction" \
    --checkpoint "scaled18000=$BASELINE" \
    --checkpoint "scaled_full=$CANDIDATE" \
    --device "$DEVICE" \
    --bootstrap "$BOOTSTRAP" \
    "${limit_args[@]}" \
    --out "$out" \
    "$@" >> "$LOG" 2>&1
  test -s "$out"
  test -s "${out%.json}_summary.json"
  log "finished ${name} ${direction}"
}

log "waiting for full-data training report"
wait_for_training

test -s "$TRAIN_REPORT"
test -s "$CANDIDATE"
test -s "$BASELINE"
test -s "$CONFIG"
test -s "$TOKENIZER/merges.txt"
test -s "$BENCHMARK_DIR/newstest2014.en-fr.jsonl"
test -s "$BENCHMARK_DIR/newstest2014.fr-en.jsonl"

if [ "$TRAIN_REPORT" -ot "$CANDIDATE" ]; then
  log "refusing stale training report: $TRAIN_REPORT is older than $CANDIDATE"
  exit 1
fi

PYTHONIOENCODING=utf-8 "$PY" - "$CANDIDATE" "$BASELINE" <<'PY' | tee -a "$LOG"
import sys
import torch

for label, path in (("candidate", sys.argv[1]), ("baseline", sys.argv[2])):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    print(f"{label}: step={payload.get('global_step')} path={path}")
    if label == "baseline" and int(payload.get("global_step", 0)) != 18000:
        raise SystemExit("baseline checkpoint is not step 18,000")
PY

log "evaluating candidate $(realpath "$CANDIDATE") against scaled step 18,000"
run_eval greedy en-fr --beam-size 1
run_eval greedy fr-en --beam-size 1
run_eval beam4 en-fr --beam-size 4 --length-penalty 1.2 --min-new-tokens 16
run_eval beam4 fr-en --beam-size 4 --length-penalty 1.2 --min-new-tokens 16

log "building direct comparison"
PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 "$PY" scripts/compare_full_vs_scaled.py \
  --reports-dir "$REPORTS_DIR" \
  --tag "$TAG" \
  --candidate "$CANDIDATE" \
  --baseline "$BASELINE" \
  --samples "$BOOTSTRAP" \
  --out "$OUT" \
  --markdown "$MARKDOWN" 2>&1 | tee -a "$LOG"

log "full-data evaluation finished: $MARKDOWN"
