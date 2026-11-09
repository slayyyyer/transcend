#!/usr/bin/env bash
# Run the 800M-token continuation and its direct 350M comparison.
#
# Training is delegated to extend_full_data.sh so the existing preservation
# and resume logic stays authoritative. When that process exits, this wrapper
# evaluates the preserved 350M best, the continuation's dev-selected best and
# the continuation's final checkpoint on identical full newstest2014 rows.
set -euo pipefail

PY="${PYTHON:-/c/Users/justin/AppData/Local/Python/pythoncore-3.14-64/python.exe}"
CONFIG="${CONFIG:-configs/scaled_full.yaml}"
TOKENIZER="${TOKENIZER:-models/cpu_small/tokenizer}"
OUTPUT_DIR="${OUTPUT_DIR:-models/scaled_full}"
REPORT="${REPORT:-$OUTPUT_DIR/training_report.json}"
PRESERVED_BEST="${PRESERVED_BEST:-$OUTPUT_DIR/best.pre_extend.pt}"
BEST="${BEST:-$OUTPUT_DIR/best.pt}"
FINAL="${FINAL:-$OUTPUT_DIR/checkpoint.pt}"
BENCHMARK_DIR="${BENCHMARK_DIR:-data/prepared/newstest2014}"
REPORTS_DIR="${REPORTS_DIR:-reports}"
TAG="${TAG:-scaled_full_800m}"
LOG="${LOG:-reports/scaled_full_800m.log}"
EVAL_LOG="${EVAL_LOG:-reports/scaled_full_800m_eval.log}"
OUT="${OUT:-$REPORTS_DIR/newstest2014.${TAG}.comparisons.json}"
MARKDOWN="${MARKDOWN:-$REPORTS_DIR/newstest2014.${TAG}.comparisons.md}"
BOOTSTRAP="${BOOTSTRAP:-1000}"
EVAL_SETTLE_S="${EVAL_SETTLE_S:-60}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$REPORTS_DIR" "$(dirname "$LOG")" "$OUTPUT_DIR"
: > "$LOG"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"
}

test -s "$CONFIG"
test -s "$TOKENIZER/merges.txt"
test -s "$BENCHMARK_DIR/newstest2014.en-fr.jsonl"
test -s "$BENCHMARK_DIR/newstest2014.fr-en.jsonl"

if [ "$DRY_RUN" = "1" ]; then
  log "dry run: would run the 800M continuation, then evaluate full350, full800best and full800final"
  log "outputs: $OUT and $MARKDOWN"
  exit 0
fi

if [ "$SKIP_TRAIN" != "1" ]; then
  log "starting 800M continuation"
  WAIT=0 SETTLE_S=0 bash scripts/extend_full_data.sh 2>&1 | tee -a "$LOG"
  log "800M continuation process exited"
else
  log "SKIP_TRAIN=1; using existing continuation artifacts"
fi

test -s "$REPORT"
test -s "$PRESERVED_BEST"
test -s "$BEST"
test -s "$FINAL"

STATE="$("$PY" - "$REPORT" "$PRESERVED_BEST" "$BEST" "$FINAL" <<'PY'
import json
import sys
from pathlib import Path
import torch

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
def step(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload.get("global_step", 0))

print(
    int(report["global_step"]),
    int(report["tokens_seen"]),
    str(report.get("stop_reason")),
    step(sys.argv[2]),
    step(sys.argv[3]),
    step(sys.argv[4]),
)
PY
)"
read -r REPORT_STEP REPORT_TOKENS STOP_REASON BASE_STEP BEST_STEP FINAL_STEP <<< "$STATE"

for value in "$REPORT_STEP" "$REPORT_TOKENS" "$BASE_STEP" "$BEST_STEP" "$FINAL_STEP"; do
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    log "could not read numeric training/checkpoint state: $STATE"
    exit 1
  fi
done

log "continuation state: report_step=$REPORT_STEP tokens=$REPORT_TOKENS stop=$STOP_REASON baseline_step=$BASE_STEP best_step=$BEST_STEP final_step=$FINAL_STEP"
if (( REPORT_TOKENS <= 350004440 )); then
  log "refusing comparison: continuation did not add training tokens"
  exit 1
fi
if (( REPORT_STEP <= 40113 )) || (( FINAL_STEP <= 40113 )); then
  log "refusing comparison: continuation did not advance beyond step 40,113"
  exit 1
fi
if (( BASE_STEP != 38000 )); then
  log "refusing comparison: preserved baseline is step $BASE_STEP, expected 38,000"
  exit 1
fi

log "allowing CUDA memory to settle for ${EVAL_SETTLE_S}s"
sleep "$EVAL_SETTLE_S"
: > "$EVAL_LOG"

run_eval() {
  local name="$1"
  local direction="$2"
  shift 2
  local out="$REPORTS_DIR/newstest2014.${TAG}.${name}.${direction}.full.json"
  log "starting ${name} ${direction} -> ${out}"
  PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 "$PY" -m nmt.cli evaluate \
    --config "$CONFIG" \
    --tokenizer "$TOKENIZER" \
    --benchmark "$BENCHMARK_DIR/newstest2014.${direction}.jsonl" \
    --direction "$direction" \
    --checkpoint "full350=$PRESERVED_BEST" \
    --checkpoint "full800best=$BEST" \
    --checkpoint "full800final=$FINAL" \
    --device cuda \
    --bootstrap "$BOOTSTRAP" \
    --out "$out" \
    "$@" >> "$EVAL_LOG" 2>&1
  test -s "$out"
  test -s "${out%.json}_summary.json"
  log "finished ${name} ${direction}"
}

run_eval greedy en-fr --beam-size 1
run_eval greedy fr-en --beam-size 1
run_eval beam4 en-fr --beam-size 4 --length-penalty 1.2 --min-new-tokens 16
run_eval beam4 fr-en --beam-size 4 --length-penalty 1.2 --min-new-tokens 16

TITLE="Full-data 800M continuation vs 350M checkpoint on newstest2014"
if (( REPORT_TOKENS < 800000000 )); then
  TITLE="Full-data continuation (${REPORT_TOKENS} tokens) vs 350M checkpoint on newstest2014"
fi

log "building direct comparison"
PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 "$PY" scripts/compare_800m.py \
  --reports-dir "$REPORTS_DIR" \
  --tag "$TAG" \
  --baseline-label full350 \
  --candidates full800best full800final \
  --baseline-step 38000 \
  --training-report "$REPORT" \
  --title "$TITLE" \
  --samples "$BOOTSTRAP" \
  --out "$OUT" \
  --markdown "$MARKDOWN" 2>&1 | tee -a "$LOG"

log "800M comparison finished: $MARKDOWN"
