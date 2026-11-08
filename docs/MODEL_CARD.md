# Model card — `transcend-enfr-accel-pilot`

## Model identity

| Field | Value |
| --- | --- |
| Name | `transcend-enfr-accel-pilot` |
| Checkpoint | `models/accel_pilot/best.pt`, optimizer step 14,500 |
| Checkpoint SHA-256 | `cf813c77caed6e7e6785318bfa8630feb70d5f26ec2887f45735a2859c126231` |
| Latest checkpoint | `models/accel_pilot/checkpoint.pt`, step 15,897 (evaluated on newstest2014 after selection; last validation at 15,500) |
| Scaled experiment checkpoint | `models/scaled_pilot/best.pt`, step 18,000 (19,825,920 parameters; separately exported, not the default) |
| Scaled checkpoint SHA-256 | `2c2f518b1f3b015fdce0cff6ef762a4a0f081643add9ec6865856ba821f22c77` |
| Scaled latest checkpoint | `models/scaled_pilot/checkpoint.pt`, step 18,799 |
| Scaled export | `models/scaled_export` (12 files, 237,211,722 B; step 18,000; not the default package) |
| Scaled export encoder SHA-256 | `8937c77c0caac9db4eba9502ebb3acf6c4414dfdea11dc42f2cf4047ed836b7f` |
| Exported package | `models/accel_export` (11 files, 120 MB; default, rebuilt from step 14,500; step-9,500 copy at `models/accel_export_step9500`) |
| Final verification package | `models/accel_export_step15897` (11 files, 120 MB; step 15,897; not the default package) |
| Default export encoder SHA-256 | `49a5b975af1681dc1a3ffb791669e312cff0701fc21d54a3a42828df2d99ffa2` |
| Languages | English ↔ French, one shared model, target-language control token |
| Training code version | repository state at 2026-11-08 |

The model was initialized from the documented random scheme `N(0, 0.02)` with
seed 1337 (`nmt/model.py::INITIALIZATION_MANIFEST`). No pretrained weights,
embeddings, tokenizer, or distilled/teacher-generated data were used at any
point.

## Architecture

- Encoder–decoder Transformer, Pre-LN, GELU, sinusoidal positions supplied as an
  explicit `position_ids` input (no baked-in shapes), tied input/output embeddings.
- 4 encoder + 4 decoder layers, `d_model` 256, 4 attention heads, FFN 1024,
  dropout 0.1, LayerNorm ε 1e-5, maximum 512 positions.
- **9,421,824 trainable parameters** (counted from the constructed model, not
  estimated).
- Joint byte-level BPE vocabulary of 8,000: ids 0–255 raw bytes, 256 word
  boundary, 257 PAD, 258 UNK, 259 BOS, 260 EOS, 261 `<2en>`, 262 `<2fr>`, merges
  from 263. Source format `<2xx> … EOS`; decoder starts at BOS.
- Limits: 128 source tokens and 136 generated tokens including controls. Inputs
  beyond the source limit are rejected, never truncated.

A separate scaled experiment uses 6 encoder + 6 decoder layers, `d_model` 320,
5 heads, FFN 1,280 and **19,825,920 trainable parameters**, with the same 8,000
byte-BPE tokenizer, positions, normalization and sequence limits. Its step-18,000
checkpoint is exported separately at `models/scaled_export`; the default
deployable package remains the 9.4M step-14,500 model.

## Training

| Field | Value |
| --- | --- |
| Data | Europarl v7 FR–EN + WMT14 News Commentary v9 (train split only) |
| Corpus scale | 150,000 sampled pairs of 565,475 prepared train pairs |
| Target tokens seen | 82,000,043 run total (74,807,994 at the step-14,500 checkpoint) |
| Optimizer updates | 15,897 (best development checkpoint at 14,500) |
| Batch | 4,096 non-padding target tokens per update, accumulated microbatches |
| Device / precision | CUDA (RTX 3050 Ti Laptop, 4 GB) / bf16 autocast |
| Wall clock | 3,761.9 s cumulative: 643.5 s initial + 1,800.2 s resumed + 1,318.2 s token-cap continuation |
| Stop | `max_train_tokens` at step 15,897 / 82,000,043 tokens |
| Throughput | 24,529 target tokens/s in the final run; 214 ms mean optimizer update |

Hyperparameters: label smoothing 0.1, AdamW (0.9, 0.98, ε 1e-9, wd 0.01),
clip norm 1.0, peak LR 5e-4, 400 warmup updates, inverse-sqrt decay, 500-update
validation interval, checkpoint every 500 updates. Gradients were normalized by
non-padding target tokens, including the final partial accumulation group.

### Scaled experiment

The scaled run used the full 565,475-pair prepared train split (not the 150,000
pair sample), an 8,192-token update budget, peak LR 4e-4, 800 warmup updates and
1,000-update validation/checkpoint intervals under a fresh 4-hour / 164M-token /
20,000-update budget. Its first invocation stopped on `max_walltime` at step
14,218; a resume from that checkpoint stopped on `max_train_tokens` at step
18,799.

| Field | Value |
| --- | --- |
| Parameters | 19,825,920 (6+6 layers, `d_model` 320, 5 heads, FFN 1,280) |
| Data | all 565,475 prepared train pairs configured; same tokenizer and limits |
| Target tokens seen | 164,002,778 |
| Optimizer updates | 18,799 |
| Best development checkpoint | step 18,000, mean chrF++ 52.6313 |
| Wall clock | 18,343.24 s cumulative (14,400.24 s first invocation + 3,943.0 s resume) |
| Stop | `max_train_tokens` at step 18,799 |
| Throughput | 10,148.42 target tokens/s in the resume; 935.21 ms mean update |

The best checkpoint is `models/scaled_pilot/best.pt` (step 18,000); the latest is
`models/scaled_pilot/checkpoint.pt` (step 18,799). The best validation was
52.9095 chrF++ EN→FR and 52.3530 FR→EN at step 18,000. The step-14,000 and
step-14,218 artifacts are preserved as `best_step14000.pt` and
`checkpoint_step14218.pt`. The scaled run changes both model size and data
exposure relative to the small pilot, so it is not a single-factor ablation.

Development metric (mean chrF++ over the two directions, 300 rows per direction
from the held-out block split; small-model history in
`models/accel_pilot/history.json`, scaled history in
`models/scaled_pilot/history.json`):

| Step | Mean chrF++ | EN→FR | FR→EN |
| --- | --- | --- | --- |
| 500 | 6.70 | 8.75 | 4.65 |
| 2,000 | 27.09 | 26.79 | 27.38 |
| 3,500 | 36.11 | 36.39 | 35.84 |
| 5,000 | 40.50 | 40.41 | 40.59 |
| 6,500 | 42.76 | 42.50 | 43.03 |
| 8,000 | 43.95 | 44.32 | 43.59 |
| 9,000 | 44.41 | 44.24 | 44.58 |
| 9,500 | 44.64 | 44.75 | 44.53 |
| 10,000 | 44.99 | 45.30 | 44.68 |
| 11,000 | 45.42 | 45.47 | 45.37 |
| 12,000 | 45.79 | 45.62 | 45.96 |
| 13,000 | 45.66 | 45.71 | 45.61 |
| 14,000 | 46.77 | 46.85 | 46.69 |
| 14,500 | **47.18** | 47.51 | 46.85 |
| 15,000 | 46.81 | 46.86 | 46.76 |
| 15,500 | 46.82 | 47.03 | 46.60 |

## Evaluation

Metrics are corpus-level SacreBLEU (`13a`, mixed case, exp smoothing,
SacreBLEU 2.5.1) and chrF++ (`word_order=2`), computed on detokenized output
with greedy decoding unless a row is labeled otherwise.

External benchmark: newstest2014 (`wmt14/full`). The FR→EN row is the same
bilingual set with sides swapped — it is not a separately published official
reverse benchmark. Segments whose source exceeds the model's limit are dropped
and counted in the report, never truncated. EN→FR: n = 2,959, 44 skipped;
FR→EN: n = 2,922, 81 skipped.

| System | EN→FR BLEU | EN→FR chrF++ | FR→EN BLEU | FR→EN chrF++ |
| --- | --- | --- | --- | --- |
| Source-copy baseline | 2.36 | 21.88 | 2.38 | 23.72 |
| Step 2,000 checkpoint | 1.37 | 19.13 | 1.45 | 19.39 |
| Step 9,500 checkpoint | 9.21 | 35.56 | 9.60 | 35.94 |
| Step 14,500 candidate (82M-token run), greedy | **11.03** | **38.67** | **11.28** | **38.86** |
| Step 15,897 final (not dev-selected), greedy | 12.11 | 39.46 | 12.17 | 39.53 |
| Step 14,500 tuned beam 4 (LP 1.2, min 16) | 13.42 | 40.82 | 13.15 | 40.88 |
| Step 15,897 final, tuned beam 4 (LP 1.2, min 16) | 14.46 | 41.37 | 14.11 | 41.45 |
| Scaled 19.8M step 14,000 (archived), greedy | 18.92 | 45.73 | 17.86 | 45.42 |
| Scaled 19.8M step 14,000, tuned beam 4 (LP 1.2, min 16) | 20.67 | 47.22 | 19.41 | 46.92 |
| Scaled 19.8M step 18,000, greedy | 20.01 | 47.04 | 19.02 | 46.70 |
| Scaled 19.8M step 18,000, tuned beam 4 (LP 1.2, min 16) | **21.24** | **48.44** | **20.30** | **48.15** |

The tuned beam rows use beam 4, length penalty 1.2 and a minimum of 16 new
target tokens, selected on dev only. For the step-14,500 row, paired chrF++
gains versus greedy are +2.15 [1.96, 2.34] EN→FR and +2.03 [1.81, 2.26] FR→EN;
BLEU gains are +2.39/+1.87. Sentence-level chrF++ there still regresses on
1,017 EN→FR and 984 FR→EN sentences, and 1,519/1,365 outputs are shorter. These
are automatic metrics, not a human evaluation; default decoding remains greedy. The final step-15,897 checkpoint
was evaluated after selection and improves on the step-14,500 candidate by
+0.79/+0.67 chrF++ (95% CIs [0.58, 1.01] and [0.47, 0.88]); it was not used to
choose the exported model.

With the same beam settings, the final checkpoint scores 14.455/41.365 EN→FR and
14.113/41.452 FR→EN, improving on final greedy by +2.345/+1.944 BLEU and
+1.906/+1.924 chrF++ (95% CIs [1.684, 2.125] and [1.734, 2.117]). Sentence
chrF++ W/L/T is 1,786/1,045/128 and 1,729/1,033/160; 2,834/2,764 outputs changed.
See `reports/newstest2014.82m.final_beam4.comparisons.md`. This is an automatic
metric comparison, not a human evaluation.

The scaled 19.8M step-18,000 checkpoint was evaluated on the same full retained
segments with the same greedy and beam-4 settings. Greedy: 20.011/47.036 EN→FR
and 19.018/46.698 FR→EN. Beam 4: 21.243/48.442 and 20.300/48.152. Versus the
final small model, greedy gains are +7.901/+6.849 BLEU and +7.577/+7.170 chrF++
(95% CIs [7.264, 7.896] and [6.863, 7.490]); beam gains are +6.788/+6.187 BLEU
and +7.077/+6.700 chrF++ ([6.785, 7.373] and [6.424, 6.982]). On the scaled
model, beam 4 improves over greedy by +1.232/+1.282 BLEU and +1.406/+1.454
chrF++ ([1.206, 1.610] and [1.280, 1.625]). The resumed checkpoint improves on
archived step 14,000 by +1.087/+1.160 BLEU and +1.307/+1.283 chrF++ greedy, and
+0.573/+0.886 BLEU and +1.226/+1.237 chrF++ beam 4. These are automatic metrics,
not human evaluation; see `reports/newstest2014.scaled_resumed.comparisons.md`.

### API-assisted spot check

The current deterministic 25-row-per-direction sample uses the final step-15,897
checkpoint for both greedy and beam-4 decoding (the same dev-selected settings
as step 14,500). It was translated through the no-key Google `gtx` endpoint and
compared by `nmt.cli review-api`. This is machine-assisted review, not a human
evaluation, and it is not a full-set comparator.

| System (25 rows/direction, final step 15,897) | EN→FR BLEU | EN→FR chrF++ | FR→EN BLEU | FR→EN chrF++ |
| --- | --- | --- | --- | --- |
| Final greedy | 3.413 | 30.176 | 18.341 | 43.487 |
| Final beam 4 (dev-tuned) | 5.383 | 32.249 | 21.780 | 45.658 |
| Google `gtx` | 31.149 | 56.128 | 49.329 | 68.008 |

The API won all 50 sentence chrF++ comparisons against both final model
variants. That is an expected result for a production translator and is not
evidence that the pilot is usable. BLEU on 25 segments is unstable. The Google
`gtx` endpoint is undocumented, has no SLA, and is suitable only for spot
checks. Reports are in `reports/review_api_google_82m_final_enfr/` and
`reports/review_api_google_82m_final_fren/`; the earlier step-14,500 comparison
remains in `reports/review_api_google_82m_enfr/` and
`reports/review_api_google_82m_fren/`. MyMemory was probed first but returned
exact newstest references on the sample (BLEU 100); it is translation-memory-
backed and is not used as an independent baseline. For this proof of concept,
Google `gtx` is the external machine-assisted comparator; it is evaluation-only
and is not used as training data.

Paired document bootstrap, 1,000 resamples over 176 documents:

| Comparison | EN→FR ΔchrF++ [95% CI] | FR→EN ΔchrF++ [95% CI] |
| --- | --- | --- |
| Step 9,500 vs step 2,000 | +16.44 [15.89, 16.97] | +16.55 [15.85, 17.21] |
| Step 9,500 vs source-copy | +13.69 [12.95, 14.44] | +12.22 [11.43, 12.99] |
| Step 14,500 vs step 9,500 | +3.10 [2.88, 3.31] | +2.92 [2.65, 3.16] |
| Step 14,500 vs source-copy | +16.79 [15.99, 17.64] | +15.13 [14.35, 15.93] |
| Step 15,897 vs step 14,500 | +0.79 [0.58, 1.01] | +0.67 [0.47, 0.88] |
| Step 15,897 vs source-copy | +17.58 [16.87, 18.34] | +15.81 [15.04, 16.61] |
| Scaled step 18,000 vs final step 15,897, greedy | +7.58 [7.26, 7.90] | +7.17 [6.86, 7.49] |
| Scaled step 18,000 vs final step 15,897, beam 4 | +7.08 [6.79, 7.37] | +6.70 [6.42, 6.98] |
| Scaled step 18,000 beam 4 vs scaled greedy | +1.41 [1.21, 1.61] | +1.45 [1.28, 1.63] |

The step-9,500 checkpoint beat both baselines on chrF++ in both directions; the
step-14,500 candidate then improved on step 9,500 by +3.10/+2.92 chrF++ and beat
source-copy by +16.79/+15.13. The final step-15,897 checkpoint improved on the
candidate by +0.79/+0.67 chrF++ and on source-copy by +17.58/+15.81; all
intervals exclude zero. The scaled step-18,000 checkpoint further improves on
the final small model by +7.58/+7.17 chrF++ greedy and +7.08/+6.70 chrF++ beam
4; those paired intervals also exclude zero. These are measured improvements
within the pilot, not evidence of usable translation quality.

Human evaluation: **not performed.** This proof of concept uses machine-assisted
comparisons only. The scaled checkpoint received automatic metrics only; it was
not included in the packet review or the Google `gtx` spot check. An AI-assisted
review of all 200 packet rows from the step-14,500 checkpoint found 43/200
acceptable (3 OK, 40 MINOR), 107 MAJOR and 50 CRITICAL; the per-row record is in
`reports/review_packet_82m_enfr/review_en-fr_reviewed.tsv` and
`reports/review_packet_82m_fren/review_fr-en_reviewed.tsv`, with aggregate results
in `reports/review_summary_82m.md`. The same deterministic 200 IDs were reviewed
for both checkpoints: 19 became acceptable and 9 stopped being acceptable, but
the fresh review also assigns more CRITICAL labels (50 vs 25), so this is not a
uniform quality gain. This is not a human evaluation: no qualified bilingual
reviewer has scored the packet, no inter-annotator agreement exists, and no
human judgment is reported. Round-trip similarity is not used.

## Runtime verification

The default export is step 14,500. The scaled step-18,000 checkpoint is exported
separately to `models/scaled_export` and checked without replacing the default
package. The final step-15,897 checkpoint remains at
`models/accel_export_step15897`.

- Scaled Python↔ONNX Runtime parity: worst max-absolute logit difference 6.44e-6
  and token-for-token identical greedy generation (`reports/parity_scaled.json`).
- Scaled Java Maven suite: 10 tests, 0 failures, 0 skipped, including the
  Python↔Java greedy parity vectors (`reports/java_build_scaled.log`).
- Scaled Java benchmark: cold load 1,542 ms; first translation 508 ms; warm
  median 354 ms, p95 562 ms; batch of 5 inputs 1,731 ms (2.89 inputs/s). See
  `reports/java_benchmark_scaled.json`.
- Final step-15,897 parity: worst logit difference 6.44e-6 and identical greedy
  generation (`reports/parity_step15897.json`); 10 Java tests passed.

## Intended use and out-of-scope use

- Intended: research and engineering demonstration of a fully reproducible,
  from-scratch training → ONNX → Java pipeline on consumer hardware.
- **Not intended for production translation.** The pilot is small, reaches the
  token cap but not the 20,000-update target, and produces omissions, invented
  morphology and repeated phrases. For example, "The train leaves at eight in
  the morning." becomes "La formation est en train de se détourner." Do not use
  it for legal, medical, safety, financial, or customer-facing content.
- Out of scope: documents, HTML, speech, language detection, languages other
  than EN/FR, and domain adaptation beyond parliamentary/news text.

## Limitations and risks

- The small run reached the 82M-token cap at 15,897 updates but not the
  20,000-update target; only 150,000 of 565,475 prepared pairs were sampled.
- Greedy external BLEU is 11.03/11.28 for the default step-14,500 package, and
  corpus chrF++ is 38.67/38.86. The final small checkpoint measured 12.11/12.17
  and 39.46/39.53; the scaled step-18,000 checkpoint measured 20.01/19.02 and
  47.04/46.70. Tuned beam improves these automatic scores further but remains
  non-production quality. Outputs can be fluent-looking but wrong. BLEU/chrF++
  are not "percent correct".
- Development chrF++ (47.18 at step 14,500) is in-domain and above the external
  result; it was used for checkpoint selection and is not a general-quality
  claim.
- Training data is parliamentary/news text; conversational and informal register
  is underrepresented, as are modern named entities.
- Splits are opaque 200-pair blocks, not verified document boundaries; duplicates
  were removed per side across splits but the limitation is documented in
  `docs/DATA_CARD.md`.
- Beam search is implemented and tested; default decoding is greedy. A dev-only
  sweep selected beam 4, length penalty 1.2 and minimum 16 new tokens. On full
  newstest2014 it raises BLEU by +2.39 EN→FR/+1.87 FR→EN and chrF++ by
  +2.15/+2.03 versus greedy, but sentence-level chrF++ regresses on 1,017/984
  rows and 1,519/1,365 outputs are shorter. See
  `reports/beam_tuned_step14500.md` and the earlier untuned sample in
  `reports/beam_vs_greedy_step14500.md`. This is not a human evaluation.
- The scaled run stopped at the 164M-token cap after 18,799 updates, not at the
  20,000-update cap. It used the full prepared split and 19.8M parameters,
  changing both size and data exposure. Its automatic scores are stronger, and
  the step-18,000 checkpoint is now exported and Java-verified, but it has no
  human review.
- No quantified bias, toxicity, or privacy audit was performed. Parliamentary
  proceedings contain personal names and political content inherited from the
  source corpora.
- No qualified human review was performed; this proof of concept uses
  machine-assisted comparisons only. A 25-row-per-direction no-key Google `gtx`
  spot check was run, but it is not a full-set comparator and its output is not
  human judgment. COMET was not run. The AI-assisted review of the step-14,500
  packet found only 43/200 acceptable (21.5%) and 50 CRITICAL rows. It is not a
  substitute for qualified human evaluation.

## Provenance, licensing and artifacts

Training corpora: Europarl v7 and WMT14 News Commentary v9 from statmt.org, with
pinned SHA-256 checksums and retrieval dates in
`data/prepared/europarl_nc/manifest.json`. Evaluation: newstest2014, recorded in
`data/prepared/newstest2014/manifest.json`. The corpora are not redistributed
here; see `NOTICE`, `LICENSE` and `docs/DATA_CARD.md`. Source-code licensing is
separate from corpus and model-artifact terms; the model artifact inherits the
corpus constraints and is published for research use only.

Raw corpora and large checkpoints are kept outside git: `data/raw/`,
`data/prepared/`, `models/`. Exact paths and hashes are listed in
`artifacts/README.md`.
