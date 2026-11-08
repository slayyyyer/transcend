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

## Next data expansion (planned, not run)

The current prepared train split uses Europarl stride 5: 565,475 train pairs
from 2,007,723 Europarl source lines plus 183,784 News Commentary lines. The
Europarl archive already on disk therefore contains roughly 1.6M additional
aligned pairs that have never been trained on. That is the cheapest,
lowest-licensing-risk expansion, and it is the recommended first quality run.

Phase 1, no new corpus download:

* Re-prepare with all Europarl v7 pairs (stride 1) plus the existing News
  Commentary v9 candidates, retaining exact deduplication, length-ratio
  filtering and the existing opaque block split. Expect roughly 2.0-2.2M pairs
  after cleaning.
* Keep the frozen 8,000-entry tokenizer and the 128/136 sequence limits fixed,
  so the first experiment changes data quantity only.
* Train the 6+6 scaled architecture from scratch for about 350M target tokens
  (about 43k updates at 8,192 tokens), keeping the same peak LR and a 12-hour
  walltime cap. At the measured 10,148 target tokens/s this is about 9.6 hours
  before validation.
* A 350M-token pass draws roughly 1.2M pairs at the observed ~290 target
  tokens/pair, more than doubling the unique-pair coverage of the 565k-pair
  run even though it is not a full pass over 2.2M pairs.
* Evaluate on the same full newstest2014 rows and 1,000-document bootstrap, and
  report the comparison to step 18,000. This is automatic metrics only.

Phase 2, only if phase 1 justifies more work:

* DGT v2021 is available as a 315 MB EN-FR archive with about 5.7M pairs and
  explicit EC re-use terms. Sample and deduplicate it (EU legal boilerplate
  repeats heavily), and treat the domain shift separately from the Europarl
  expansion.
* WMT News Commentary v18.1 (46 MB, about 0.3M pairs) is available, but it is
  the highest-risk source for newstest2014 contamination; it needs exact and
  n-gram overlap filtering before use.
* Do not start with ParaCrawl or OpenSubtitles at this scale. They are hundreds
  of millions of noisy pairs, expensive to clean on this machine, and their
  redistribution terms are unclear or incompatible with the artifact policy.
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
  `reports/newstest2014.scaled.comparisons.json`.
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
