# Decisions and constraints

This file records the choices that shape the project, including the ones forced
by the actual machine and the ones that remain unresolved.

## Hardware actually used

Inspected on the project machine (`reports/hardware.json`):

| Item | Value |
| --- | --- |
| OS | Windows 11 (10.0.26300), x64 |
| CPU | 12th Gen Intel Core (16 logical threads) |
| RAM | 16 GB total (~8.3 GiB available at inspection) |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop, 4096 MiB, compute capability 8.6 |
| Disk | 81 GB free on `C:` |
| Python | 3.14.6, PyTorch 2.14.1+cu130 |
| Java | Temurin 21.0.13 (installed locally under `.tools/`) |

No hardware was purchased, rented or provisioned. The CUDA build of PyTorch was
already installed; the CPU path is the reference path and is fully supported.

## Architecture

* Encoder-decoder Transformer, Pre-LN, GELU FFN, sinusoidal positions.
* Tied input/output embeddings (joint vocabulary).
* Explicit additive masks; explicit `position_ids` inputs; explicit KV cache.
* Random initialization: N(0, 0.02) matrices, zero biases, LayerNorm 1/0, from a
  dedicated `torch.Generator` seeded with the experiment seed. No pretrained
  weights, embeddings or tokenizer are loaded at any point.
* Attention is written with `matmul`/`softmax` rather than a fused kernel. This
  is deliberate: it exports to ONNX with a documented cache layout, and it is
  easy to audit.

Presets (layer counts are encoder+decoder):

| preset | layers | d_model | heads | FFN | vocab | parameters (measured) |
| --- | --- | --- | --- | --- | --- | --- |
| smoke | 2+2 | 128 | 4 | 512 | 1,015 (fixture-limited) | 1,056,128 |
| cpu_small | 2+2 | 128 | 4 | 512 | 8,000 | 1,950,208 |
| existing_accelerator | 4+4 | 256 | 4 | 1024 | 8,000 | 9,421,824 |
| scaled_accelerator | 6+6 | 320 | 5 | 1280 | 8,000 | 19,825,920 |
| compact | 4+4 | 384 | 6 | 1536 | (would need 32k vocab) | not run |

The 16k/32k vocabularies were not trained. Tokenizer training on the full train
split took 2m40s for 8,000 entries; spending several more minutes on a larger
vocabulary would have taken time away from optimizer updates. The same
8,000-entry artifact is used by all trained models, which also makes their
results directly comparable.

## Tokenizer

A **byte-level BPE with explicit word boundaries** is used instead of
SentencePiece. Reasons:

1. The deployed Java library must reproduce token ids exactly. Integer-only BPE
   over a merge-rank table is bit-identical in both languages without a JNI
   bridge; SentencePiece would have required either a native binding or a
   reimplementation with subtle normalization differences.
2. Every UTF-8 byte is in the base vocabulary (ids 0..255), so there is no UNK
   path and no byte-fallback divergence to prove.
3. Encoding per word is cacheable, which keeps CPU preprocessing cheap.

Trade-offs, stated plainly: no learned linguistic normalization, and sequences
are longer than a well-trained 32k SentencePiece model would produce. The
tokenizer artifact (`merges.txt` + `tokenizer.json`) is frozen and shipped with
the model.

Normalization policy: Unicode NFC, then runs of the six ASCII whitespace
characters collapse to one space, then ASCII whitespace is stripped from both
ends. Non-ASCII spaces (NBSP) are preserved as content. Both the Python and Java
implementations follow exactly this policy, and both are tested against the same
golden token ids.

Source layout: `<2xx>` target-language control id + source tokens + EOS.
Target layout: target tokens + EOS; decoder input is BOS + target tokens.
Control ids are never produced by text encoding, so user text cannot change the
requested direction.

## Sequence limits

Measured coverage with the 8,000-entry tokenizer:

| limit | English | French |
| --- | --- | --- |
| 64 tokens | ~65% | ~58% |
| 96 tokens (chosen for `cpu_small`) | 89.1% | 84.7% |
| 128 tokens (accelerator preset) | ~95% | ~93% |

`cpu_small` therefore starts at 96 source tokens, not the initial 64. This is the
documented "growth after measurement" allowed by the brief. Over-length training
pairs are filtered with counts recorded in `run_manifest.json`; over-length
inference input is rejected with an exception rather than truncated.

## Data and splits

* Europarl v7 FR-EN and WMT14 News Commentary v9, downloaded from statmt.org,
  SHA-256 pinned in `data/prepared/europarl_nc/manifest.json`.
* Deterministic stride sampling across each release (Europarl stride 5,
  News Commentary stride 1), then cleaning and exact deduplication:
  584,796 raw pairs -> 576,091 kept (1,475 empty, 1,319 identical, 5,364 exact
  duplicates, 547 length-ratio rejects).
* 565,475 train / 5,321 dev / 5,098 test pairs, assigned by hashing an **opaque
  contiguous block id** (block size 200), not a claimed document id. Both
  releases are distributed as flat line-aligned text without document metadata;
  this limitation is recorded rather than hidden.
* Cross-split leakage removal drops any train pair whose English or French side
  also occurs in dev/test; assertions in `contamination_report.json` fail the
  build if any overlap remains.
* Final external benchmark: newstest2014 (`wmt14/full`, 3,003 segments,
  176 documents). Both directions come from official WMT14 files in the same
  archive. newstest2013 is **not** used: the statmt.org WMT13 archive contains
  source files only, with no parallel English/French reference pair.

## Rights

Neither Europarl v7 nor News Commentary v9 carries a standard permissive
licence in this repository's assessment. OPUS lists Europarl's licence as
unknown; WMT distribution does not grant blanket commercial or redistribution
rights; the news articles remain their publishers' copyright. The project
therefore records research/development use only and does **not** assert rights to
redistribute the raw corpora or the derived model weights. Fixtures under
`data/fixtures/` are original and CC0. Source code is separate from corpus and
artifact terms; see `NOTICE`.

## Precision, devices, resumption

* CPU training is FP32. CUDA training uses BF16 autocast (the device reports
  BF16 support), with FP32 master weights and no gradient scaler.
* Checkpoints store model/optimizer/scaler state, Python/NumPy/Torch/CUDA RNG
  state, sampler position, token accounting, hashes and history. The analytic
  learning-rate schedule is reconstructed from the global step, so no separate
  scheduler object exists. They load with `map_location="cpu"` and are migrated
  explicitly.
* `--max-walltime` is a per-invocation budget. A resumed run starts a fresh
  clock, while the checkpoint records cumulative wall time. A regression test
  covers a checkpoint whose prior wall time already exceeds the new limit.
* CPU determinism is tested: a 30-update run resumed at step 15 is compared
  weight-for-weight against an uninterrupted 30-update run and must match
  exactly. Accelerator runs are not expected to be bit-identical.
* MPS was unavailable on this machine and is therefore untested.

## Training runs

* Initial run: 2,000 updates, 10,313,701 target tokens, 643.5 s, best development
  chrF++ 27.09. It was stopped manually after overrunning its walltime inside a
  validation pass; the step-2,000 report is reconstructed and archived.
* Resumed run: 7,635 updates, 39,351,844 target tokens, 1,800.2 s (30 minutes),
  best development chrF++ 44.64 at step 9,500. It stopped on `max_walltime` at
  step 9,635. Cumulative: 2,443.7 s (~40.7 minutes), 49,665,545 target tokens.
* Token-cap continuation: 6,262 updates, 32,334,498 target tokens, 1,318.2 s,
  best development chrF++ 47.18 at step 14,500. It stopped on
  `max_train_tokens` at step 15,897 / 82,000,043 target tokens. Cumulative:
  3,761.9 s (~62.7 minutes), 82,000,043 target tokens.
* Measured throughput on the RTX 3050 Ti (BF16 autocast): 21,860 target
  tokens/s in the resumed run and 24,529 target tokens/s in the final
  continuation; 214 ms mean optimizer update in the final run, with 93.3 s in
  validation and 2.8 s in checkpointing.
* The `--max-walltime` accounting was corrected after the first session: it is
  now a per-invocation budget, so a resume starts a fresh local clock. A
  regression test covers a checkpoint whose cumulative wall time already exceeds
  the new per-run limit.
* The 20,000-step target was not reached: realized groups average about 5,158
  non-padding target tokens per update, so the 82M-token cap arrives first.
  Batching was not changed mid-run. The unused 20,000-step headroom is not a
  quality guarantee.
* Scaled run: `configs/scaled_accelerator.yaml`, 19,825,920 parameters, the full
  565,475-pair prepared train split (no 150k sample), 8,192-token update budget,
  and fresh 164M-token / 20,000-update / 4-hour caps. The first invocation
  stopped on `max_walltime` at step 14,218 / 123,987,566 target tokens /
  14,400.24 s. A resume stopped on `max_train_tokens` at step 18,799 /
  164,002,778 target tokens, with 3,943.0 s this invocation and 18,343.24 s
  cumulative. The best development checkpoint was step 18,000 at 52.6313 mean
  chrF++; `best.pt` is that checkpoint and `checkpoint.pt` is step 18,799.
  Resumed throughput was 10,148.42 target tokens/s and 935.21 ms per update.
  The step-14,000 and step-14,218 artifacts are preserved as
  `best_step14000.pt` and `checkpoint_step14218.pt`.

## Data expansion: full Europarl (phase 1 complete)

The current prepared train split uses Europarl stride 5: 565,475 train pairs
from 2,007,723 Europarl source lines plus 183,784 News Commentary lines. The
Europarl archive already on disk therefore contains roughly 1.6M additional
aligned pairs that have never been trained on. That is the cheapest,
lowest-licensing-risk expansion, and it is the recommended first quality run.

Phase 1, no new corpus download:

* Re-prepared with all Europarl v7 pairs (stride 1) plus all existing News
  Commentary v9 candidates, retaining exact deduplication, length-ratio
  filtering and the existing opaque block split. The result is
  `data/prepared/europarl_full`: 2,190,974 raw candidates, 2,143,227 kept,
  and 2,101,321 train / 20,138 dev / 19,789 test pairs. Cross-split overlap is
  zero in the contamination report.
* The frozen 8,000-entry tokenizer and the 128/136 sequence limits are fixed,
  so this experiment changes data quantity only relative to the scaled run.
* Trained the 6+6 scaled architecture from scratch for at most 350M target
  tokens (about 43k updates at 8,192 tokens), keeping the same peak LR and a
  12-hour walltime cap. The run completed in `models/scaled_full` with
  `configs/scaled_full.yaml`: 40,113 updates, 350,004,440 target tokens,
  33,528.56 s (9h 19m), stopping on `max_train_tokens`. Realized throughput was
  10,438.99 target tokens/s. The best development checkpoint was step 38,000 at
  54.8099 mean chrF++; the final checkpoint was step 40,113. Evaluation used the
  dev-selected `best.pt`, not the final checkpoint.
* The training manifest records 3,991,640 directional examples and 219,771,441
  target tokens in one epoch. A 350M-token budget is therefore about 1.6 epochs
  over 2,101,321 train pairs, so all 2,009,059 usable pairs are seen rather
  than a 1.2M-pair subset.
* Evaluated on the same full newstest2014 rows and 1,000-document bootstrap, and
  reported against scaled step 18,000. This is automatic metrics only.
* `scripts/extend_full_data.sh` continued the same model from its 350M-token
  checkpoint to an 800M-token / 100k-update cumulative budget, with a 24-hour
  per-invocation walltime and patience 30. It preserved the 350M report,
  `best.pt`, `checkpoint.pt` and history under `.pre_extend` names, then resumed
  from `checkpoint.pt`; it did not reinitialize or restart the run. The token and
  step limits are cumulative, while `--max-walltime` applies afresh to each
  invocation. The learning-rate schedule remains the existing inverse-square-
  root schedule.

### 800M continuation result (complete)

The continuation resumed at step 88,000 / 767,805,099 tokens and stopped on
`max_train_tokens` at step 91,691 / 800,002,117 tokens, adding 3,691 updates and
32,197,018 tokens in 1,907.66 s. Cumulative reported walltime is 64,152.22 s.
The dev-selected best remains step 77,000 at 56.0833 mean chrF++; the final
step-91,691 checkpoint last validated at 55.8353. The preserved 350M baseline
is step 38,000, not the final 350M checkpoint.

On the same retained newstest2014 rows (2,959 EN→FR / 2,922 FR→EN), the final
800M checkpoint scores 25.367/51.391 EN→FR and 23.558/50.830 FR→EN greedy, and
26.440/52.531 and 24.377/51.902 beam 4. Against the 350M baseline the BLEU gains
are +2.712/+2.298 greedy and +2.361/+2.232 beam 4; chrF++ gains are
+2.346/+2.075 and +2.213/+2.020. The paired chrF++ 95% intervals are
[2.106, 2.587], [1.859, 2.282], [1.998, 2.451] and [1.801, 2.253]; all exclude
zero. The 77,000 best is also above baseline, and the final test scores are
slightly higher than the best, but the final-vs-best difference was not
bootstrap-tested. This is a positive but diminishing-return result at roughly
2.3x the token budget, not evidence of production quality or human acceptability.

Phase 2, after the extension and data-expansion arms:

* DGT v2021 is available as a 315 MB EN-FR archive with about 5.7M pairs and
  explicit EC re-use terms. Sample and deduplicate it (EU legal boilerplate
  repeats heavily), and treat the domain shift separately from the Europarl
  expansion.
* WMT News Commentary v18.1 (46 MB, about 0.3M pairs) is available, but it is
  the highest-risk source for newstest2014 contamination; it needs exact and
  n-gram overlap filtering before use.
* OpenSubtitles, GlobalVoices, WikiMatrix and Giga-French remain later
  candidates. ParaCrawl is now added only as a bounded, filtered sample; see
  "Everyday-language expansion" below for the measured limits and licensing
  caveats.
* More data is not a guaranteed BLEU gain. The 4 GB card limits model capacity,
  and the current run is still far below compute-optimal tokens per parameter;
  the phase-1 experiment is designed to test the data constraint cleanly.

## Evaluation protocol

* Development decisions use the in-domain Europarl dev split (300 rows per
  direction for validation). The beam sweep used a deterministic every-17th-row
  dev sample (313 rows per direction; 305 EN→FR and 296 FR→EN retained) and did
  not touch newstest2014 for selection.
* The final test set (newstest2014) is used for checkpoint comparison, not
  hyperparameter tuning. Full-set comparisons use 1,000 paired bootstrap
  resamples over the 176 documents. The step-9,500 checkpoint was compared with
  the archived step-2,000 checkpoint and the raw-text source-copy baseline
  (`reports/newstest2014.comparisons.json`); the step-14,500 checkpoint was
  compared directly with step 9,500 (`reports/newstest2014.82m.comparisons.json`).
  The final step-15,897 checkpoint was evaluated after selection against the
  step-14,500 candidate and source-copy
  (`reports/newstest2014.82m.final.comparisons.json`). The scaled step-18,000
  checkpoint was evaluated after its token-cap stop against the final small
  greedy and beam-4 runs on exactly the same retained newstest2014 segments
  (`reports/newstest2014.scaled_resumed.comparisons.json`); the archived
  step-14,000 comparison remains in
  `reports/newstest2014.scaled.comparisons.json`. The full-data comparison is
  generated by `scripts/evaluate_full.sh`, which waits for
  `models/scaled_full/training_report.json`, evaluates
  `models/scaled_full/best.pt` against scaled step 18,000 on the same retained
  rows, and writes `reports/newstest2014.scaled_full.comparisons.{json,md}` via
  `scripts/compare_full_vs_scaled.py`. The completed comparison is positive in
  all four evaluations: greedy 22.655/49.045 EN→FR and 21.260/48.755 FR→EN;
  beam 4 24.079/50.318 and 22.145/49.882. Against step 18,000, the chrF++ gains
  are +2.009 [1.742, 2.302], +2.057 [1.783, 2.310], +1.876 [1.633, 2.113] and
  +1.730 [1.496, 1.977]; the BLEU gains are +2.644/+2.242 greedy and
  +2.836/+1.845 beam 4. Every paired 95% interval excludes zero. This is
  automatic metrics only. The 800M continuation comparison is in
  `reports/newstest2014.scaled_full_800m.comparisons.{json,md}`: the final
  step-91,691 checkpoint is above the preserved step-38,000 baseline in all
  four evaluations by +2.712/+2.298 BLEU and +2.346/+2.075 chrF++ greedy, and
  +2.361/+2.232 BLEU and +2.213/+2.020 chrF++ beam 4. Every paired chrF++
  95% interval excludes zero; final-vs-best was not tested.
* Metrics: SacreBLEU (`13a`, cased) and chrF++ (`word_order=2`), with signatures.
* A documented 564-row EN→FR subset retains random-initialized, CPU-pilot and
  accelerator baselines (`reports/subset600.en-fr*`). The full-set baselines are
  source-copy and the step-2,000 checkpoint. A COMET comparator was **not** run:
  it would require downloading model weights. A small API-assisted spot check
  was run instead; it is not a full-set comparator and is not human review.
* Human evaluation: **not performed.** This proof of concept uses machine-assisted
  comparisons only. An AI-assisted review of all 200 packet rows
  from the step-14,500 checkpoint found 43 acceptable (3 OK, 40 MINOR), 107 MAJOR
  and 50 CRITICAL. The same deterministic IDs were reviewed for step 9,500:
  19 became acceptable and 9 stopped being acceptable, but the fresh review also
  assigns more CRITICAL labels (50 vs 25), so this is not a uniform quality gain.
  This is not a qualified human evaluation: a single AI reviewer scored the
  packet, no inter-annotator agreement exists, and the reviewer was not blinded
  to the model. Per-row results are in `reports/review_packet_82m_enfr/` and
  `reports/review_packet_82m_fren/`; aggregates are in
  `reports/review_summary_82m.{json,md}`.
* Beam decoding: the existing implementation collapsed to greedy at the first
  step because it expanded identical BOS rows before top-k, and it dropped
  completed EOS hypotheses. Both defects were fixed; `--beam-size`,
  `--length-penalty` and `--min-new-tokens` are exposed by `nmt.cli evaluate`,
  and evaluation uses a batched beam path (single-sentence parity is tested).
  Dev tuning selected beam 4, length penalty 1.2, minimum 16 new tokens;
  LP 1.5 over-lengthened and lost BLEU, while minimum 8 had no effect. Full
  newstest2014 gives 13.417/40.817 EN→FR and 13.147/40.881 FR→EN, versus greedy
  11.031/38.667 and 11.278/38.855. Paired chrF++ gains are +2.150 [1.961, 2.337]
  and +2.026 [1.805, 2.263]; sentence-level chrF++ W/L/T is 1,835/1,017/107 and
  1,797/984/141. This is mixed at sentence level, not a human evaluation; greedy
  remains the default. See `reports/beam_tuning_dev.md` and
  `reports/beam_tuned_step14500.md`; the earlier untuned 495-row sample is in
  `reports/beam_vs_greedy_step14500.md`. The same settings on the final
  step-15,897 checkpoint give 14.455/41.365 EN→FR and 14.113/41.452 FR→EN,
  versus final greedy 12.110/39.459 and 12.169/39.528. Paired chrF++ gains are
  +1.907 [1.684, 2.125] and +1.924 [1.734, 2.117]; sentence W/L/T is
  1,786/1,045/128 and 1,729/1,033/160. See
  `reports/newstest2014.82m.final_beam4.comparisons.md`. The scaled step-18,000
  checkpoint scores 20.011/47.036 EN→FR and 19.018/46.698 FR→EN greedy, and
  21.243/48.442 and 20.300/48.152 beam 4. Against the final small model, beam
  gains are +6.788/+6.187 BLEU and +7.077/+6.700 chrF++ (95% CIs [6.785, 7.373]
  and [6.424, 6.982]); on the scaled model, beam over greedy adds +1.406/+1.454
  chrF++ ([1.206, 1.610] and [1.280, 1.625]). See
  `reports/newstest2014.scaled_resumed.comparisons.md`. This is automatic metrics
  only.
* API-assisted review (`nmt/review_api.py`): the current deterministic
  25-row-per-direction sample uses the final step-15,897 checkpoint for both
  greedy and beam-4 decoding (the same dev-selected settings as step 14,500).
  It is translated by a no-key Google `gtx` endpoint and compared using
  sentence/corpus chrF++ and BLEU. On the 50 sampled rows the API scored BLEU
  31.149/49.329 and chrF++ 56.128/68.008 (EN→FR/FR→EN), versus final beam-4
  5.383/21.780 and 32.249/45.658 and final greedy 3.413/18.341 and 30.176/43.487;
  the API wins all 50 sentence chrF++ comparisons. This is a
  machine-assisted sanity check, not a human evaluation, and BLEU on 25 segments
  is unstable. The endpoint is undocumented and has no SLA. MyMemory was probed
  first but returned exact newstest references (BLEU 100) because it is
  translation-memory-backed; it remains optional via `--provider mymemory` with
  an explicit caveat and is not treated as an independent baseline. Google `gtx`
  is the external machine-assisted comparator for this proof of concept; it is
  evaluation-only. The earlier step-14,500 gtx reports remain under
  `reports/review_api_google_82m_enfr/` and `reports/review_api_google_82m_fren/`.

## Everyday-language expansion (Tatoeba and bounded ParaCrawl)

This iteration adds data-pipeline support only. No model was trained on the new
sources, so none of the measured BLEU/chrF++ results in this repository are
claims about Tatoeba or ParaCrawl.

* Tatoeba is acquired from the weekly per-language sentence files plus the
  EN-FR link file. The weekly files are UTF-8 and do not contain a per-row
  license or review column; the source-level terms are CC BY 2.0 FR by default,
  with some rows also CC0 1.0. Pair IDs retain both sentence IDs for
  attribution. All translations of one English sentence share a split block.
* ParaCrawl v9 EN-FR is acquired only as a bounded compressed prefix, 64 MiB by
  default. The inspected TXT has two tab-separated columns and **no** per-row
  Bicleaner score; the parser supports an optional score column but does not
  pretend one exists here. The sample is a prefix, not a uniform sample of the
  21 GiB release.
* The new-source cleaner rejects malformed markup/URLs/control characters,
  wrong-language rows (Tatoeba tags; a conservative heuristic for ParaCrawl),
  excessive repetition, and unreasonable length mismatches. The smoke sample
  retained 440,351 of 442,283 Tatoeba pairs and 61,152 of 83,151 ParaCrawl
  pairs from an 8 MiB prefix. Those are sample statistics, not corpus
  estimates.
* Mixtures are built on top of an existing prepared baseline. The baseline
  splits are copied verbatim; new sources are capped independently, split by
  their own groups, deduplicated against every baseline split, and checked for
  exact and near-duplicate contamination. Per-source caps are configurable, so
  a large ParaCrawl sample cannot exceed its configured share.
* Near-duplicate contamination uses word 3-gram Jaccard at a configurable
  threshold (default 0.85). Exact overlap assertions remain mandatory. The
  smoke run removed zero near-duplicates and had zero exact overlaps; this
  validates the guard, not the full-corpus leakage rate.
* Controlled comparisons keep the same architecture, seed, frozen 8,000-entry
  tokenizer, decoding settings and 350M-token budget. A new tokenizer, if one is
  ever trained, must read `train.tsv` only; an existing checkpoint must not be
  retokenized.
* The four configs are `configs/scaled_full.yaml` (existing data),
  `configs/scaled_full_tatoeba.yaml`, `configs/scaled_full_paracrawl.yaml`, and
  `configs/scaled_full_combined.yaml`. Exact preparation and training commands
  are in `docs/DATA_CARD.md`; the long runs are deliberately not launched.
* The separate 24-row everyday review set covers questions, negation, names,
  numbers, idioms and *tu/vous*. Its French references are project-authored
  drafts and are labeled `not_bilingual_human_reviewed`; it is excluded from
  training and is not a general-quality benchmark.
* OpenSubtitles, GlobalVoices, WikiMatrix and Giga-French remain later
  candidates. The full ParaCrawl release is not a first-step download.

## Acceptance criteria recorded before final testing

Engineering criteria (must pass): export/parity, deterministic mask/cache
behaviour, bounded generation, resume, Java/Python token and output equality.
These pass: 34 Python tests, 10 Java tests against both the step-14,500 and
step-15,897 packages, ONNX parity with worst logit absolute differences of
8.58e-6 and 6.44e-6, and identical greedy generation.

Quality criteria for this hardware budget (expected, not guaranteed): the
accelerator pilot should beat source-copy on chrF++ in both directions and show
recognisable word-level transfer on frequent vocabulary. The step-9,500
checkpoint beat source-copy (+13.69 EN→FR, +12.22 FR→EN chrF++, with 95%
document-bootstrap intervals excluding zero); the step-14,500 checkpoint then
beat step 9,500 by +3.10/+2.92 chrF++ and source-copy by +16.79/+15.13. The
final step-15,897 checkpoint improved on the candidate by +0.79/+0.67 chrF++
and on source-copy by +17.58/+15.81; all intervals exclude zero. The scaled
step-18,000 checkpoint then improved on the final small model by +7.58/+7.17
chrF++ greedy and +7.08/+6.70 chrF++ beam 4; those paired intervals also exclude
zero.

General-purpose translation quality is explicitly **not** claimed; BLEU is
11.03/11.28 for the candidate, 12.11/12.17 for the final small checkpoint, and
20.01/19.02 for the scaled greedy checkpoint, and sample sentences are
frequently wrong. The AI-assisted sentence review of the
step-14,500 packet found 21.5% acceptable; the fresh review's distribution is
not uniformly better (50 CRITICAL rows vs 25 archived). Qualified human review
was not performed; this proof of concept uses machine-assisted comparisons only.
