# transcend — English↔French neural machine translation, trained from scratch

A reproducible EN↔FR translation project whose tokenizer and translation model are
both trained from scratch, plus an offline Java 17 inference
library that runs the exported ONNX package without Python or network access.

**Status: proof of concept; pipeline complete, pilot reached the 82M-token cap
at 15,897 updates, quality measured — still not a usable translator.** Every
stage below actually ran end to end. The step-14,500 checkpoint improves on step
9,500 on newstest2014 in both directions: BLEU 11.03/11.28 and chrF++
38.67/38.86, with paired chrF++ gains of +3.10 EN→FR and +2.92 FR→EN (95% CIs
exclude zero). The final step-15,897 checkpoint, evaluated after selection,
raises the greedy scores to BLEU 12.11/12.17 and chrF++ 39.46/39.53. A dev-tuned
beam-4 decoder (length penalty 1.2, minimum 16 tokens) raises the full-set
automatic scores to BLEU 13.42/13.15 and chrF++ 40.82/40.88; greedy remains the
default. A fresh AI-assisted sentence-level review of the step-14,500 packet
found 43/200 acceptable, but 157/200 MAJOR or CRITICAL,
and ordinary sentences are frequently mistranslated (see [Measured
results](#measured-results)). This demonstrates a working, auditable training →
export → Java pipeline, not translation quality.

## What "from scratch" means here

- The tokenizer is a byte-level BPE trained only on this project's training split.
  Its vocabulary is built from scratch: ids 0–255 are raw bytes, 256 is the word
  boundary, 257–262 are PAD/UNK/BOS/EOS/`<2en>`/`<2fr>`, merges start at 263.
  There is no SentencePiece model, no pretrained vocabulary, and no JNI: Python
  and Java implement the same merge algorithm, verified by golden vectors.
- The model is a hand-written Pre-LN Transformer encoder–decoder built from
  PyTorch primitives (`nmt/model.py`), initialized from a seeded
  `N(0, 0.02)` scheme recorded in `INITIALIZATION_MANIFEST`. It does not
  instantiate `MarianMTModel`, load any checkpoint, or download any weights.
- No distillation, no teacher-generated data, no back-translation from an
  external model, and no remote translation at training or inference time.
- `nmt/smoke.py` proves the whole path (random init → forward/backward →
  generation → ONNX → parity) on an original fixture corpus.

## Repository layout

```
nmt/                 Python training, evaluation and export package
  data/              acquisition, cleaning, block splits, contamination audit
  model.py           Transformer, masks, seeded initialization manifest
  tokenizer.py       byte-level BPE (Python reference implementation)
  train.py           bounded training loop, resume, validation, checkpoints
  decode.py          greedy + beam search with KV-cache reordering
  export_onnx.py     encoder / decoder-init / decoder-step graphs + checksums
  parity.py          Python ↔ ONNX Runtime numeric and generation parity
  evaluate.py        SacreBLEU + chrF++ end to end + paired bootstrap
  review_packet.py   blind bilingual review packet generator
  review_api.py      machine-assisted greedy/beam vs translation API comparison
java/                Maven module: ONNX Runtime Java inference library + CLI
configs/             architecture/recipe presets (smoke, cpu_small,
                     existing_accelerator, compact)
scripts/             fixture generation, Java golden vectors, baseline helpers
data/                acquisition and prepared manifests (raw archives git-ignored)
models/              tokenizer, pilot checkpoints, exported packages (git-ignored)
reports/             hardware, training, evaluation, parity and benchmark JSON
docs/                decisions, data card, model card
```

## Quick start

Use a Python 3.10+ interpreter. The commands below invoke it as `$PY`; set that
variable first:

```bash
PY=python  # replace with the full path to your Python 3.10+ interpreter if needed
```

```bash
# 1. Record the actual hardware (CPU, RAM, GPU, disk, framework versions)
$PY -m nmt.cli hardware --out reports/hardware.json

# 2. Acquire and prepare corpora (bounded; writes manifests and a contamination report)
$PY -m nmt.cli prepare --source all --out data/prepared --max-pairs 300000

# 3. Train the joint tokenizer on the train split only
$PY -m nmt.cli tokenizer --train-dir data/prepared/europarl_nc --out models/cpu_small/tokenizer --vocab-size 8000

# 4. Prove the end-to-end path on an original fixture corpus
$PY -m nmt.cli smoke --out models/smoke/smoke_report.json

# 5. Train (or resume) with explicit limits; stops at the first reached limit.
#    --max-walltime is a budget for this invocation, not a lifetime cap.
$PY -m nmt.cli train --config configs/existing_accelerator.yaml \
  --prepared data/prepared/europarl_nc --tokenizer models/cpu_small/tokenizer \
  --out models/accel_pilot --device cuda --max-steps 20000 --max-train-tokens 82000000 --max-walltime 2400

$PY -m nmt.cli train --config configs/existing_accelerator.yaml \
  --prepared data/prepared/europarl_nc --tokenizer models/cpu_small/tokenizer \
  --out models/accel_pilot --resume models/accel_pilot/checkpoint.pt --device cuda \
  --max-steps 20000 --max-train-tokens 82000000 --max-walltime 2400

# 6. Evaluate both directions on the pinned external benchmark, comparing the
#    archived step-9,500 checkpoint with the current candidate.
$PY -m nmt.cli evaluate --config configs/existing_accelerator.yaml \
  --tokenizer models/cpu_small/tokenizer \
  --benchmark data/prepared/newstest2014/newstest2014.en-fr.jsonl --direction en-fr \
  --checkpoint step9500=models/accel_pilot/best_step9500.pt \
  --checkpoint candidate=models/accel_pilot/best.pt \
  --device cuda --bootstrap 1000 --out reports/newstest2014.82m.en-fr.full.json

$PY -m nmt.cli evaluate --config configs/existing_accelerator.yaml \
  --tokenizer models/cpu_small/tokenizer \
  --benchmark data/prepared/newstest2014/newstest2014.fr-en.jsonl --direction fr-en \
  --checkpoint step9500=models/accel_pilot/best_step9500.pt \
  --checkpoint candidate=models/accel_pilot/best.pt \
  --device cuda --bootstrap 1000 --out reports/newstest2014.82m.fr-en.full.json

# 7. Export the ONNX package, then check Python ↔ ONNX Runtime parity
$PY -m nmt.cli export --config configs/existing_accelerator.yaml \
  --checkpoint models/accel_pilot/best.pt --tokenizer models/cpu_small/tokenizer \
  --out models/accel_export --provenance data/prepared/europarl_nc/manifest.json
$PY -m nmt.cli parity --model-dir models/accel_export --out reports/parity.json

# Export the final step-15,897 checkpoint separately for verification.
$PY -m nmt.cli export --config configs/existing_accelerator.yaml \
  --checkpoint models/accel_pilot/checkpoint.pt --tokenizer models/cpu_small/tokenizer \
  --out models/accel_export_step15897 --provenance data/prepared/europarl_nc/manifest.json
$PY -m nmt.cli parity --model-dir models/accel_export_step15897 \
  --out reports/parity_step15897.json

# 8. Build blind bilingual review packets from the step-14,500 reports.
#    These create packets; they do not add human review results.
$PY -m nmt.cli review-packet --report reports/newstest2014.82m.en-fr.full.json \
  --out reports/review_packet_82m_enfr --per-direction 100
$PY -m nmt.cli review-packet --report reports/newstest2014.82m.fr-en.full.json \
  --out reports/review_packet_82m_fren --per-direction 100

# 9. Optional machine-assisted spot check against a free, no-key API.
#    This is not human review and does not replace the blind packets.
$PY -m nmt.cli review-api --provider google-gtx \
  --greedy-report reports/newstest2014.82m.en-fr.full.json \
  --beam-report reports/beam_tuned_step14500.en-fr.full.json \
  --out reports/review_api_google_82m_enfr --per-direction 25
$PY -m nmt.cli review-api --provider google-gtx \
  --greedy-report reports/newstest2014.82m.fr-en.full.json \
  --beam-report reports/beam_tuned_step14500.fr-en.full.json \
  --out reports/review_api_google_82m_fren --per-direction 25
```

Java (JDK 17+ and Maven; configure your installation if they are not on `PATH`):

```bash
cd java
mvn -B package -Dtranscend.model.dir=../models/accel_export  # unit + parity tests
CP="target/classes:target/lib/*"  # use the classpath separator for your platform
java -cp "$CP" org.transcend.nmt.Main translate \
  --model ../models/accel_export --from en --to fr "The train leaves at eight in the morning."
java -cp "$CP" org.transcend.nmt.Main bench \
  --model ../models/accel_export --from en --to fr --count 20 --out ../reports/java_benchmark.json
```

The exported package (`manifest.json`, `model_config.json`, `generation.json`,
`tokenizer/tokenizer.json`, `tokenizer/merges.txt`, three ONNX graphs,
`parity_vectors.json`, provenance and checksums) is self-contained.
`Translator.open` verifies every checksum, loads the graphs once, and performs no
downloads, training, subprocess calls or fallback. The local
`models/accel_export/` package was rebuilt from the step-14,500 checkpoint; the
previous step-9,500 package is archived at
`models/accel_export_step9500/`. The final step-15,897 checkpoint is exported
separately at `models/accel_export_step15897/` for evaluation and runtime
verification; it is not the default package. Each package has new checksums,
and ONNX parity and the Java suite were rerun against it.

## Architecture and training recipe

| Item | `existing_accelerator` (main run) | `cpu_small` (pilot only) |
| --- | --- | --- |
| Layers (enc+dec) | 4 + 4 | 2 + 2 |
| `d_model` / heads / FFN | 256 / 4 / 1024 | 128 / 4 / 512 |
| Parameters (actual) | 9,421,824 | 1,950,208 |
| Vocabulary | 8,000 joint byte-BPE | 8,000 |
| Positions | sinusoidal, explicit `position_ids` | same |
| Normalization | Pre-LN, GELU, tied embeddings | same |
| Source / target limit | 128 / 136 tokens (incl. controls) | 96 / 104 |

Recipe: label-smoothed cross-entropy 0.1 with padding ignored, AdamW
(β=0.9/0.98, ε=1e-9, wd=0.01), gradient clipping 1.0, linear warmup then
inverse-square-root decay (peak 5e-4), 4,096 non-padding target tokens per
optimizer update accumulated from measured microbatches, dynamic padding with
length bucketing. Gradients are normalized by real non-padding target tokens,
including the final partial group; the scheduler advances per optimizer update.
Checkpoints store model/optimizer/scaler state, RNG, sampler position, data
accounting, tokenizer and data hashes, and validation history. The learning-rate
schedule is analytic, so the global step is the scheduler state; there is no
separate scheduler object to persist. CPU resume is weight-for-weight identical
to an uninterrupted run (covered by tests). `--max-walltime` is a budget for the
current invocation: a resumed run gets a fresh local limit, while cumulative time
is recorded in the checkpoint.

The actual pilot ran to the 82M-token cap: 2,000 updates in 643.5 s, a
7,635-update continuation in 1,800.2 s (30 minutes), and a 6,262-update
continuation in 1,318.2 s, for 15,897 updates / 82,000,043 non-padding target
tokens / 3,761.9 s (62.7 minutes) of accelerator time. The best development
checkpoint is step 14,500 (47.18 mean chrF++); the final resumable checkpoint is
step 15,897. Throughput in the final run was 24,529 target tokens/s with BF16
autocast. The run stopped on the token cap before 20,000
updates because realized groups average about 5,158 target tokens rather than
exactly 4,096; batching was not changed mid-run.

## Measured results

External benchmark: newstest2014 (`wmt14/full`), greedy decoding unless noted,
detokenized output, SacreBLEU 2.5.1 (`13a`, cased; chrF++ `word_order=2`).
EN→FR uses 2,959 segments (44 overlength sources skipped); FR→EN uses 2,922
segments (81 skipped). Skipped segments are recorded in the reports and are never
truncated.

| System | EN→FR BLEU | EN→FR chrF++ | FR→EN BLEU | FR→EN chrF++ |
| --- | --- | --- | --- | --- |
| Source-copy baseline | 2.36 | 21.88 | 2.38 | 23.72 |
| Step 2,000 checkpoint | 1.37 | 19.13 | 1.45 | 19.39 |
| Step 9,500 checkpoint | 9.21 | 35.56 | 9.60 | 35.94 |
| Step 14,500 candidate (82M-token run), greedy | **11.03** | **38.67** | **11.28** | **38.86** |
| Step 15,897 final (not dev-selected), greedy | 12.11 | 39.46 | 12.17 | 39.53 |
| Step 14,500 tuned beam 4 (LP 1.2, min 16) | 13.42 | 40.82 | 13.15 | 40.88 |

The tuned beam row is the full-set comparison from
`reports/beam_tuned_step14500.md`: beam 4, length penalty 1.2, minimum 16 new
tokens, selected on dev only. Paired chrF++ gains versus greedy are +2.15
[1.96, 2.34] EN→FR and +2.03 [1.81, 2.26] FR→EN; BLEU gains are +2.39/+1.87.
Results remain mixed at the sentence level: 1,017/984 sentences regress on
sentence chrF++ and 1,519/1,365 outputs are shorter. This is an automatic metric
comparison; default decoding remains greedy.

The final step-15,897 checkpoint was evaluated after selection; it improves on
the step-14,500 candidate by +0.79/+0.67 chrF++ (95% CIs [0.58, 1.01] and
[0.47, 0.88]) and was not used to choose the exported model. See
`reports/newstest2014.82m.final.md`.

Paired document bootstrap, 1,000 resamples over 176 documents. The final
checkpoint and source-copy comparisons are in
`reports/newstest2014.82m.final.comparisons.json` (Markdown:
`reports/newstest2014.82m.final.md`); the step-14,500-vs-step-9,500 comparison
is in `reports/newstest2014.82m.comparisons.json`, and the two prior step-9,500
comparisons are archived in `reports/newstest2014.comparisons.json`:

| Comparison | EN→FR ΔchrF++ [95% CI] | FR→EN ΔchrF++ [95% CI] |
| --- | --- | --- |
| Step 9,500 vs step 2,000 | +16.44 [15.89, 16.97] | +16.55 [15.85, 17.21] |
| Step 9,500 vs source-copy | +13.69 [12.95, 14.44] | +12.22 [11.43, 12.99] |
| Step 14,500 vs step 9,500 | +3.10 [2.88, 3.31] | +2.92 [2.65, 3.16] |
| Step 14,500 vs source-copy | +16.79 [15.99, 17.64] | +15.13 [14.35, 15.93] |
| Step 15,897 vs step 14,500 | +0.79 [0.58, 1.01] | +0.67 [0.47, 0.88] |
| Step 15,897 vs source-copy | +17.58 [16.87, 18.34] | +15.81 [15.04, 16.61] |

A 200-example AI-assisted packet review (100 per direction) of the **step-14,500
checkpoint** found **43/200 acceptable** (3 OK, 40 MINOR), 107 MAJOR and 50
CRITICAL. This is not a qualified human evaluation; the full rubric and per-row
records are in `reports/review_summary_82m.md` and the packet directories. Of the
same 200 rows, 19 became acceptable and 9 stopped being acceptable versus the
archived step-9,500 review. The shift is not uniform: the fresh review also
assigns more CRITICAL labels (50 vs 25), so this is not a general quality claim.

### API-assisted spot check (not human review)

A separate deterministic 25-row-per-direction sample was translated with
Google's no-key `gtx` endpoint through `nmt.cli review-api`. This is a
machine-assisted spot check, not a human evaluation, not a full-set benchmark,
and not a quality estimate for the pilot: the API is a production system and a
much stronger MT baseline, so its scores say more about the sample than about
this model.

| System (25 rows/direction) | EN→FR BLEU | EN→FR chrF++ | FR→EN BLEU | FR→EN chrF++ |
| --- | --- | --- | --- | --- |
| Greedy | 3.996 | 29.785 | 15.867 | 42.223 |
| Tuned beam 4 | 4.662 | 30.414 | 19.532 | 44.947 |
| Google `gtx` | 31.149 | 56.128 | 49.329 | 68.008 |

The API beat greedy and tuned beam on sentence chrF++ in all 50 sampled rows
(25/25 in each direction). That is expected for a production translator; it is
not a human judgment. The sample is diagnostic-stratified, and BLEU on 25
segments is unstable. Per-row TSV, summary JSON and Markdown are in
`reports/review_api_google_82m_enfr/` and `reports/review_api_google_82m_fren/`.

MyMemory was probed first, but it returned exact newstest references on the
small sample (BLEU 100), so it is not an independent baseline. It remains
available via `--provider mymemory` for retrieval-style comparison, with a
report caveat; it is translation-memory-backed. Google `gtx` is an undocumented
no-key endpoint with no SLA, suitable for spot checks only. For this proof of
concept it is the external machine-assisted comparator; it is used for
evaluation only, not as training data, and no qualified human review was
performed.

The development metric (in-domain block holdout, 300 rows per direction) rose
from 27.09 mean chrF++ at step 2,000 to **47.18** at step 14,500; the last
validation, at step 15,500, was 46.82. The final step-15,897 checkpoint was
evaluated on newstest2014 after selection, not used for model selection. The
lower external scores are the honest measure of transfer. BLEU and chrF++ are corpus-level metrics, not "percent correct", and
these results are not evidence of usable general translation.

## Data and licensing

Training data: Europarl v7 (FR–EN) plus WMT14 News Commentary v9, fetched from
statmt.org, SHA-256 pinned in `data/prepared/*/manifest.json`. Splits are opaque
contiguous 200-pair blocks hashed 98/1/1 with cross-split EN/FR overlap removal
and assertions (`contamination_report.json`). The tokenizer saw training text
only. Final evaluation uses newstest2014 (wmt14/full) in both directions; the
reverse direction is the same bilingual set with sides swapped, not a separately
published benchmark. Q4/2000 Europarl material is reserved. Segments above the
source limit are dropped and counted, never silently truncated.

Corpus terms are **not** permissive by default: Europarl's original terms allow
use but not blanket redistribution, WMT distribution does not grant every
commercial right, and this repository therefore does not invent a license for the
data. See `NOTICE`, `docs/DATA_CARD.md` and `docs/MODEL_CARD.md`. Source code is
separately licensed under `LICENSE`; model artifacts inherit the corpus
constraints recorded in the model card.

## Limits

- One sentence or short segment per call, explicit source/target language, EN→FR
  and FR→EN only. No document context, HTML, speech, or language detection.
- Inputs above the source limit are rejected with a clear error
  (`InputTooLongException` in Java, `ValueError` in Python); nothing is truncated.
- Parliamentary and news text only. Conversational and general-domain competence
  is not established.
- Beam search is implemented and tested. The dev-tuned beam-4 setting (LP 1.2,
  min 16) raises full newstest2014 BLEU/chrF++ by +2.39/+2.15 EN→FR and
  +1.87/+2.03 FR→EN versus greedy, but 1,017/984 sentences regress on sentence
  chrF++ and many outputs shorten. This is an automatic metric result, not a
  human evaluation. Default decoding remains greedy; see
  `reports/beam_tuned_step14500.md` and the earlier sample in
  `reports/beam_vs_greedy_step14500.md`.
- API-assisted review is a 25-row-per-direction spot check with an unofficial
  no-key Google endpoint. It is not a full benchmark, COMET, or human review.

## Verification actually run

- `pytest -q`: **33 passed** on CPU, covering tokenizer parity, masking, causal
  decoder behavior, padding, memorization, resume determinism, ONNX export,
  Python↔ONNX Runtime parity, beam-search validity/exploration, batched/single
  beam parity, and minimum-length EOS blocking. Review-harness tests cover
  UTF-8 chunking, provider response parsing, provider-scoped caches, dry-run
  output, and report alignment.
- `mvn -B package -Dtranscend.model.dir=../models/accel_export`: **10 tests, 0
  failures**, including the Java↔Python greedy parity tests.
- The same 10-test suite passed against
  `../models/accel_export_step15897` with **0 failures and 0 skipped**,
  including the final package's Python↔Java greedy parity test.
- Python↔ONNX parity on three cases: step-14,500 worst max-absolute logit
  difference **8.58e-6** (`reports/parity.json`); step-15,897 **6.44e-6**
  (`reports/parity_step15897.json`). Generation is token-for-token identical in
  both packages.
- Java benchmark on the step-14,500 package: cold load 1,297 ms; first
  translation 380 ms; warm median 233 ms, p95 443 ms; batch of 5 inputs
  1,219 ms (4.10 inputs/s). See `reports/java_benchmark.json`.
- Java benchmark on the step-15,897 package: cold load 1,189 ms; first
  translation 405 ms; warm median 262 ms, p95 331 ms; batch of 5 inputs
  1,436 ms (3.48 inputs/s). See `reports/java_benchmark_step15897.json`.
- Full newstest2014 evaluation in both directions for the step-15,897,
  step-14,500 and step-9,500 checkpoints, using 1,000-sample document bootstrap
  intervals. The final-checkpoint and source-copy comparisons are in
  `reports/newstest2014.82m.final.comparisons.json`; other reports are under
  `reports/`.
- Full newstest2014 tuned beam-4 (LP 1.2, min 16), both directions
  (2,959/2,922 retained): BLEU +2.39/+1.87 and chrF++ +2.15/+2.03 versus greedy;
  1,017/984 sentence-level chrF++ regressions; see
  `reports/beam_tuned_step14500.md`.
- Real API-assisted spot check: 25 rows per direction with Google `gtx`
  (40 new requests plus 10 cached responses; 50/50 usable), with per-row TSV,
  summary JSON and Markdown under `reports/review_api_google_82m_enfr/` and
  `reports/review_api_google_82m_fren/`. This is not a human evaluation.
- Historical beam-4 vs greedy sample on the first 500 raw rows per direction (495
  retained each): BLEU +2.79/+2.44 and chrF++ +1.54/+2.09, but 932/990 outputs
  changed and 575 got shorter; see `reports/beam_vs_greedy_step14500.md`.
- AI-assisted bilingual review of the step-14,500 packet: **200/200 rows
  reviewed**; 43/200 acceptable (3 OK, 40 MINOR), 107 MAJOR, 50 CRITICAL. This
  is not a qualified human evaluation; human review was not performed. See
  `reports/review_summary_82m.md`.

## Known limitations of the trained pilot

- The run reached the 82M-token cap at 15,897 updates but not the configured
  20,000 updates, because realized groups average about 5,158 target tokens;
  only 150,000 of 565,475 prepared pairs were sampled. Ordinary sentences remain
  unreliable: for example, "The train leaves at eight in the morning." becomes
  "La formation est en train de se détourner." That is a wrong translation.
- Greedy BLEU is 11.03/11.28 and chrF++ 38.67/38.86 for the step-14,500
  candidate, and 12.11/12.17 and 39.46/39.53 for the final checkpoint; both are
  far from a useful translator. Outputs show omissions, invented morphology and
  repeated phrases.
- Dev chrF++ (47.18 at step 14,500) is in-domain and higher than the external
  result; it is used for checkpoint selection, not as a quality claim.
- No qualified human review was performed; this proof of concept uses
  machine-assisted comparisons only. A 50-row no-key Google `gtx` spot check was
  run; it is not a full-set comparator and its output is not a human judgment.
  COMET was not run. The AI-assisted review of the step-14,500 packet found only
  43/200 acceptable (21.5%) and 50 CRITICAL rows. It supports, but does not
  replace, human evaluation. Round-trip translation is not treated as evidence
  of correctness.
- Splits are opaque 200-pair blocks, not verified document boundaries; leakage is
  checked per side but the limitation is documented in `docs/DATA_CARD.md`.
- Changing the tokenizer or architecture requires a fresh run. The 82M-token
  cap has been spent; further gains would require a new training budget or
  recipe, not a trivial resume. That is a plan, not a promise of useful quality.
