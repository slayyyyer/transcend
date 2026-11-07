# Model card — `transcend-enfr-accel-pilot`

## Model identity

| Field | Value |
| --- | --- |
| Name | `transcend-enfr-accel-pilot` |
| Checkpoint | `models/accel_pilot/best.pt`, optimizer step 14,500 |
| Checkpoint SHA-256 | `cf813c77caed6e7e6785318bfa8630feb70d5f26ec2887f45735a2859c126231` |
| Latest checkpoint | `models/accel_pilot/checkpoint.pt`, step 15,897 (last validation at 15,500) |
| Exported package | `models/accel_export` (11 files, 120 MB; rebuilt from step 14,500; step-9,500 copy at `models/accel_export_step9500`) |
| Export encoder SHA-256 | `49a5b975af1681dc1a3ffb791669e312cff0701fc21d54a3a42828df2d99ffa2` |
| Languages | English ↔ French, one shared model, target-language control token |
| Training code version | repository state at 2026-11-07; no git history was present |

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

Development metric (mean chrF++ over the two directions, 300 rows per direction
from the held-out block split; full history in `models/accel_pilot/history.json`):

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
| Step 14,500 tuned beam 4 (LP 1.2, min 16) | 13.42 | 40.82 | 13.15 | 40.88 |

The tuned beam row uses beam 4, length penalty 1.2 and a minimum of 16 new
target tokens, selected on dev only. Paired chrF++ gains versus greedy are +2.15
[1.96, 2.34] EN→FR and +2.03 [1.81, 2.26] FR→EN; BLEU gains are +2.39/+1.87.
Sentence-level chrF++ still regresses on 1,017 EN→FR and 984 FR→EN sentences,
and 1,519/1,365 outputs are shorter. These are automatic metrics, not a human
evaluation; default decoding remains greedy.

Paired document bootstrap, 1,000 resamples over 176 documents:

| Comparison | EN→FR ΔchrF++ [95% CI] | FR→EN ΔchrF++ [95% CI] |
| --- | --- | --- |
| Step 9,500 vs step 2,000 | +16.44 [15.89, 16.97] | +16.55 [15.85, 17.21] |
| Step 9,500 vs source-copy | +13.69 [12.95, 14.44] | +12.22 [11.43, 12.99] |
| Step 14,500 vs step 9,500 | +3.10 [2.88, 3.31] | +2.92 [2.65, 3.16] |

The step-9,500 checkpoint beat both baselines on chrF++ in both directions; the
step-14,500 checkpoint then improved on step 9,500 by +3.10/+2.92 chrF++. Those
are measured improvements within the pilot, not evidence of usable translation
quality. A fresh step-14,500-vs-source-copy bootstrap was not run; the corpus
scores are 38.67/38.86 versus 21.88/23.72.

Human evaluation: **pending**. An AI-assisted review of all 200 packet rows from
the step-14,500 checkpoint found 43/200 acceptable (3 OK, 40 MINOR), 107 MAJOR
and 50 CRITICAL; the per-row record is in
`reports/review_packet_82m_enfr/review_en-fr_reviewed.tsv` and
`reports/review_packet_82m_fren/review_fr-en_reviewed.tsv`, with aggregate results
in `reports/review_summary_82m.md`. The same deterministic 200 IDs were reviewed
for both checkpoints: 19 became acceptable and 9 stopped being acceptable, but
the fresh review also assigns more CRITICAL labels (50 vs 25), so this is not a
uniform quality gain. This is not a human evaluation: no qualified bilingual
reviewer has scored the packet, no inter-annotator agreement exists, and no
human judgment is reported. Round-trip similarity is not used.

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

- The run reached the 82M-token cap at 15,897 updates but not the 20,000-update
  target; only 150,000 of 565,475 prepared pairs were sampled.
- Greedy external BLEU is 11.03/11.28, and corpus chrF++ is 38.67/38.86. The
  tuned beam improves these automatic scores but remains non-production quality.
  Outputs can be fluent-looking but wrong. BLEU/chrF++ are not "percent correct".
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
- No quantified bias, toxicity, or privacy audit was performed. Parliamentary
  proceedings contain personal names and political content inherited from the
  source corpora.
- No qualified human review, COMET, or external pretrained MT comparator was
  run; COMET and an external comparator remain optional offline tools. The
  AI-assisted review of the step-14,500 packet found only 43/200 acceptable
  (21.5%) and 50 CRITICAL rows. It is not a substitute for qualified human
  evaluation.

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
