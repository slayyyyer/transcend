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
| compact | 4+4 | 384 | 6 | 1536 | (would need 32k vocab) | not run |

The 16k/32k vocabularies were not trained. Tokenizer training on the full train
split took 2m40s for 8,000 entries; spending several more minutes on a larger
vocabulary would have taken time away from optimizer updates. The same
8,000-entry artifact is used by both trained models, which also makes their
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
  (`reports/newstest2014.82m.final.comparisons.json`).
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
  `reports/beam_vs_greedy_step14500.md`.
* API-assisted review (`nmt/review_api.py`): a deterministic 25-row-per-direction
  sample is translated by a no-key Google `gtx` endpoint and compared with
  greedy and tuned beam using sentence/corpus chrF++ and BLEU. On the 50 sampled
  rows the API scored BLEU 31.149/49.329 and chrF++ 56.128/68.008 (EN→FR/FR→EN),
  versus tuned beam 4.662/19.532 and 30.414/44.947; the API wins all 50 sentence
  chrF++ comparisons. This is a machine-assisted sanity check, not a human
  evaluation, and BLEU on 25 segments is unstable. The endpoint is undocumented
  and has no SLA. MyMemory was probed first but returned exact newstest
  references (BLEU 100) because it is translation-memory-backed; it remains
  optional via `--provider mymemory` with an explicit caveat and is not treated
  as an independent baseline. Google `gtx` is the external machine-assisted
  comparator for this proof of concept; it is evaluation-only.

## Acceptance criteria recorded before final testing

Engineering criteria (must pass): export/parity, deterministic mask/cache
behaviour, bounded generation, resume, Java/Python token and output equality.
These pass: 33 Python tests, 10 Java tests, ONNX parity with a worst logit
absolute difference of 8.58e-6, and identical greedy generation.

Quality criteria for this hardware budget (expected, not guaranteed): the
accelerator pilot should beat source-copy on chrF++ in both directions and show
recognisable word-level transfer on frequent vocabulary. The step-9,500
checkpoint beat source-copy (+13.69 EN→FR, +12.22 FR→EN chrF++, with 95%
document-bootstrap intervals excluding zero); the step-14,500 checkpoint then
beat step 9,500 by +3.10/+2.92 chrF++ and source-copy by +16.79/+15.13. The
final step-15,897 checkpoint improved on the candidate by +0.79/+0.67 chrF++
and on source-copy by +17.58/+15.81; all intervals exclude zero.
General-purpose translation quality is explicitly **not** claimed; BLEU is
11.03/11.28 for the candidate and 12.11/12.17 for the final checkpoint, and
sample sentences are frequently wrong. The AI-assisted sentence review of the
step-14,500 packet found 21.5% acceptable; the fresh review's distribution is
not uniformly better (50 CRITICAL rows vs 25 archived). Qualified human review
was not performed; this proof of concept uses machine-assisted comparisons only.
