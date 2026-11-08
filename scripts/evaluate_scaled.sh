#!/usr/bin/env bash
# Evaluate the scaled checkpoint after training reaches its limit.
#
# Waits for the training report so the live job is never competing for the GPU,
# then runs full newstest2014 greedy and dev-tuned beam-4 in both directions.
set -euo pipefail

PY="${PYTHON:-/c/Users/justin/AppData/Local/Python/pythoncore-3.14-64/python.exe}"
TRAIN_REPORT="${TRAIN_REPORT:-models/scaled_pilot/training_report.json}"
BEST="${BEST:-models/scaled_pilot/best.pt}"
LOG="${LOG:-reports/scaled_eval.log}"
TAG="${TAG:-scaled}"
DEADLINE_S="${DEADLINE_S:-14400}"

mkdir -p reports
: > "$LOG"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"
}

wait_for_training() {
  local deadline=$((SECONDS + DEADLINE_S))
  while [ ! -f "$TRAIN_REPORT" ]; do
    if (( SECONDS >= deadline )); then
      log "timed out waiting for $TRAIN_REPORT"
      return 1
    fi
    sleep 30
  done
  # The report is written just before the training process exits. Give CUDA
  # memory a moment to be released before loading the evaluation model.
  sleep 60
}

run_eval() {
  local name="$1"
  local direction="$2"
  shift 2
  local out="reports/newstest2014.${TAG}.${name}.${direction}.full.json"
  log "starting ${name} ${direction} -> ${out}"
  PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 "$PY" -m nmt.cli evaluate \
    --config configs/scaled_accelerator.yaml \
    --tokenizer models/cpu_small/tokenizer \
    --benchmark "data/prepared/newstest2014/newstest2014.${direction}.jsonl" \
    --direction "$direction" \
    --checkpoint "scaled=${BEST}" \
    --device cuda \
    --out "$out" \
    "$@" >> "$LOG" 2>&1
  test -s "$out"
  test -s "${out%.json}_summary.json"
  log "finished ${name} ${direction}"
}

log "waiting for scaled training"
wait_for_training
test -s "$TRAIN_REPORT"
test -s "$BEST"
if [ "$TRAIN_REPORT" -ot "$BEST" ]; then
  log "refusing stale training report: $TRAIN_REPORT is older than $BEST"
  exit 1
fi
log "scaled training finished; evaluating checkpoint $(realpath "$BEST")"

run_eval greedy en-fr --beam-size 1
run_eval greedy fr-en --beam-size 1
run_eval beam4 en-fr --beam-size 4 --length-penalty 1.2 --min-new-tokens 16
run_eval beam4 fr-en --beam-size 4 --length-penalty 1.2 --min-new-tokens 16

log "all scaled evaluations finished"
