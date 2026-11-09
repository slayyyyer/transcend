# Data card — Transcend EN↔FR

## Scope and current status

This card covers the original Europarl/News Commentary corpus and the
everyday-language expansion added in this iteration: **Tatoeba** and a
bounded, heavily filtered **ParaCrawl v9** sample. The expansion was prepared
and smoke-tested; no new model was trained on it in this iteration, so no
translation-quality improvement is claimed.

The reported measured models (`models/accel_pilot`, `models/scaled_pilot`,
`models/scaled_full`) were trained only on Europarl v7 + WMT14 News Commentary
v9. The existing tokenizer remains frozen for controlled comparisons.

## Source inventory

### Existing baseline

| Field | Value |
| --- | --- |
| Sources | Europarl v7 FR–EN and WMT14 News Commentary v9 |
| Acquisition | statmt.org URLs in `nmt/data/acquire.py`, SHA-256 recorded in each prepared manifest |
| Prepared full split | 2,101,321 train / 20,138 dev / 19,789 test |
| Raw candidates | 2,190,974; 2,143,227 kept; zero exact cross-split overlap |
| Terms | Europarl research distribution; News Commentary/WMT distribution does not grant blanket redistribution. See `NOTICE`. |

### Tatoeba

| Field | Value |
| --- | --- |
| Release used | weekly export dated **2026-11-07** |
| Files | `eng_sentences.tsv.bz2`, `fra_sentences.tsv.bz2`, `eng-fra_links.tsv.bz2` |
| URLs | `https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2`; `.../fra/fra_sentences.tsv.bz2`; `.../eng/eng-fra_links.tsv.bz2` |
| Format | UTF-8 TSV. Sentences: `id<TAB>lang<TAB>text`; links: `eng_id<TAB>fra_id` |
| Compressed sizes | 24,933,870 B; 9,798,519 B; 2,675,177 B (37,407,566 B total) |
| Current SHA-256 | `1a2d75ac07aa5d035bff4e2bf053139dfd3b1f1645f6fd3d76fc852cc9ef6020`; `ac4206ab4f175af8f7e5d05bf7307679a121526599c97b6f6d8a666ab4435ef0`; `d212149f572cb6d914e1eee427bacbeace7bbea24a9709df57af4e674d5fbe3b` |
| Terms | Tatoeba sentence text is CC BY 2.0 FR by default, with some rows also available under CC0 1.0. The weekly files do not include a per-row license column. |
| Attribution | Credit Tatoeba (https://tatoeba.org) and its contributors. Tatoeba sentence IDs are retained in pair IDs so attribution can be resolved. |
| Redistribution | Raw weekly exports are not redistributed here. Derived data must respect CC BY 2.0 FR attribution; some rows may be CC0 1.0. |
| Observed shape | 442,791 links; 442,283 pairs with both sides present; 508 missing sentence IDs; 440,351 pairs passed the quality cleaner. |

Tatoeba provides sentence and link exports; it does not provide a stable
pre-generated EN–FR pair file at this path. `read_tatoeba` therefore joins the
three weekly files. All translations attached to one English sentence ID share
one split block, preserving a related-sentence group.

### ParaCrawl

| Field | Value |
| --- | --- |
| Release used | **v9**, source object last modified 2021-09-16 |
| URL | `https://web-language-models.s3.amazonaws.com/paracrawl/release9/en-fr/en-fr.txt.gz` |
| Full size | 21,863,614,702 B (about 21 GiB), **not downloaded in full** |
| Format | UTF-8 gzip, two tab-separated columns: English source and French target. No per-row quality column in this TXT release. |
| Retrieval | HTTP Range prefix; default cap **64 MiB compressed**, configurable. The compressed prefix is cached in `data/raw/paracrawl/`. |
| Terms | ParaCrawl packages the parallel data under CC0 1.0, but states that it does not own the underlying website text. |
| Attribution | Acknowledge the ParaCrawl project (https://paracrawl.eu) and cite *ParaCrawl: Web-Scale Acquisition of Parallel Corpora* (ACL 2020). |
| Observed issues | Wrong-language pairs, near-duplicate boilerplate, URLs/markup, misalignments (for example Bristol/Calvi versus Brest/Calvi), and very uneven web-domain coverage. |

The v9 TXT stream has no Bicleaner score. The parser accepts an optional third
numeric column and can filter it with `min_score`, but no score column was
present in the inspected release. Filtering is therefore heuristic; this is a
documented limitation, not a quality-score substitute.

### Evaluation-only sets

newstest2014 remains the external news benchmark. The new
`data/review/everyday_review.tsv` set is separate and excluded from training.
Its 24 French references are project-authored drafts and have **not** received
bilingual human review.

## Acquisition and reproducibility

`nmt/data/acquire.py` records, for every source:

- source version/release, owner, corpus, URL and archive name;
- license, attribution, terms URL and redistribution note;
- byte count, SHA-256 of the retrieved file or bounded sample;
- partial/full status, total source size, HTTP ETag and Last-Modified where
  available;
- retrieval time and intended use.

Tatoeba files are downloaded in full (about 37 MB). ParaCrawl uses a bounded
HTTP Range read: the default is 64 MiB compressed, and `--paracrawl-max-bytes`
changes it. The 21 GiB release is never required for a first experiment.
ParaCrawl sample checksums are run-specific because they cover the selected
prefix; the full-object ETag and size are recorded separately.

## Cleaning and filtering

The original Europarl/News Commentary cleaning is unchanged for the baseline:
Unicode NFC, ASCII-whitespace normalization, empty/identical/markup/length
checks, exact-pair deduplication, and opaque-block splitting.

For the new sources, `clean_pairs(quality_filter=True, ...)` additionally
rejects:

- malformed rows: control characters, replacement characters, HTML-ish
  markup, URLs and entity artifacts;
- wrong-language rows: Tatoeba uses its authoritative `eng`/`fra` tags;
  ParaCrawl uses a conservative English/French function-word and accent
  heuristic, with short strings left to the length and alignment checks;
- excessive repetition: long character runs, a word repeated four or more
  times, or repeated 2/3-grams;
- unreasonable length mismatch: character ratio, word ratio, and a short side
  against a much longer side;
- wrong sentence alignments that survive the ratio checks are still possible;
  ParaCrawl has no score to remove them.

Measured smoke sample (fixtures baseline, 2,000-pair caps, first 8 MiB of
ParaCrawl, 2026-11-09):

| Source | Input pairs | Kept after cleaning | Selected | Notes |
| --- | ---: | ---: | ---: | --- |
| Tatoeba | 442,283 | 440,351 | 2,000 | 1,691 repetition, 139 ratio, 81 length-mismatch, 19 identical, 2 malformed |
| ParaCrawl | 83,151 | 61,152 | 2,000 | 12,485 repetition, 9,465 language, 49 malformed |

These are smoke-sample statistics, not full-corpus estimates. Tatoeba's weekly
files change; checksums pin the run, not future weeks.

## Mixing, sampling and deduplication

`nmt.cli prepare-mixture` starts from an already prepared baseline and copies
its splits verbatim. It then adds the selected new sources:

- per-source caps are independent and configurable:
  `--tatoeba-max-pairs` and `--paracrawl-max-pairs` default to 100,000;
- Tatoeba is capped by English-sentence group, so a cap never splits all
  translations of one sentence;
- ParaCrawl is capped by a deterministic hash over the selected bounded prefix;
- the baseline recipe remains `data/prepared/europarl_full`, not a rebuilt or
  re-sampled approximation.

Pairs are deduplicated within and across all training sources using normalized,
case-folded sides with punctuation removed; reversed pairs are unified. The
baseline row wins, then Tatoeba, then ParaCrawl. The baseline dev/test sides are
treated as held-out: new training rows overlapping them are removed, and new
held-out rows overlapping baseline training are removed.

Contamination checks are run after the exact checks:

- exact pair/text overlap assertions for every train/dev/test combination;
- word 3-gram Jaccard near-duplicate checks at a configurable threshold
  (default 0.85), with common shingles capped to keep the check bounded;
- new training rows near either side of baseline dev/test, new dev/test, or
  the review set are removed;
- base training overlap with baseline held-out text is reported as an audit;
  the baseline is not silently rewritten.

The smoke mixture had **0** exact cross-split overlaps and **0** near-duplicate
training leaks. This validates the checks; it is not evidence about the full
corpus.

## Splits, grouping and leakage

- Existing baseline rows keep their existing opaque 200-pair blocks.
- Tatoeba rows are grouped by English sentence ID (`tatoeba:<eng_id>:<fra_id>`)
  so translations of one sentence stay together.
- ParaCrawl has no document identifiers in the bounded TXT prefix; contiguous
  blocks of 200 accepted rows are used. These blocks are not claimed to be
  documents.
- The review set is kept outside every split and checked against the final
  corpus by exact text. It is not used for model selection or training.

## Tokenizer policy

The existing 8,000-entry byte-level BPE tokenizer is frozen for all controlled
comparisons, including the four data arms. This isolates the data change and
avoids invalidating the continuing checkpoint.

If a new tokenizer is trained, the existing command takes `--train-dir` and
reads `train.tsv` only:

```bash
$PY -m nmt.cli tokenizer --train-dir data/prepared/<mixture> \
  --out models/tokenizer_<mixture> --vocab-size 8000
```

When continuing an existing checkpoint, do **not** retokenize and do not load a
new vocabulary. The checkpoint stores its tokenizer hash.

## Everyday review set

`data/review/everyday_review.tsv` contains 24 project-authored rows:

- 4 everyday questions;
- 4 negation examples;
- 4 names;
- 4 numbers/dates;
- 4 idioms;
- 4 formal/informal *tu/vous* pairs.

All rows carry `reference_review=not_bilingual_human_reviewed`. The generated
manifest marks `human_reviewed: false` and `reference_status:
not_bilingual_human_reviewed`. The set is excluded from training and from the
baseline splits. It is a diagnostic review set, not a statistically
representative benchmark.

## Reproducible commands

Set `PY` to the project interpreter.

```bash
# 0. Review set; this does not train anything.
$PY -m nmt.cli prepare-review \
  --source data/review/everyday_review.tsv \
  --out data/prepared/everyday_review

# 1. Rebuild the reproducible baseline (optional; existing models used the
#    already-present europarl_full directory). This writes
#    data/prepared/europarl_full.
$PY -m nmt.cli prepare --source europarl_full --out data/prepared

# 2. Individual data arms on top of that baseline. These are preparation
#    commands; they do not launch training.
$PY -m nmt.cli prepare-mixture \
  --base data/prepared/europarl_full \
  --out data/prepared/europarl_full_tatoeba \
  --include-tatoeba --tatoeba-max-pairs 100000 \
  --review-dir data/prepared/everyday_review

$PY -m nmt.cli prepare-mixture \
  --base data/prepared/europarl_full \
  --out data/prepared/europarl_full_paracrawl \
  --include-paracrawl --paracrawl-max-pairs 100000 \
  --paracrawl-max-bytes 67108864 \
  --review-dir data/prepared/everyday_review

$PY -m nmt.cli prepare-mixture \
  --base data/prepared/europarl_full \
  --out data/prepared/europarl_full_combined \
  --include-tatoeba --include-paracrawl \
  --tatoeba-max-pairs 100000 --paracrawl-max-pairs 100000 \
  --paracrawl-max-bytes 67108864 \
  --review-dir data/prepared/everyday_review

# 3. Train each arm from scratch with the same architecture, seed, frozen
#    tokenizer and 350M-token budget. Run only after reviewing the manifests.
#    (The current scaled_full.yaml is the existing-data baseline.)
$PY -m nmt.cli train --config configs/scaled_full.yaml \
  --prepared data/prepared/europarl_full \
  --tokenizer models/cpu_small/tokenizer \
  --out models/data_ablation/baseline --device cuda

$PY -m nmt.cli train --config configs/scaled_full_tatoeba.yaml \
  --prepared data/prepared/europarl_full_tatoeba \
  --tokenizer models/cpu_small/tokenizer \
  --out models/data_ablation/tatoeba --device cuda

$PY -m nmt.cli train --config configs/scaled_full_paracrawl.yaml \
  --prepared data/prepared/europarl_full_paracrawl \
  --tokenizer models/cpu_small/tokenizer \
  --out models/data_ablation/paracrawl --device cuda

$PY -m nmt.cli train --config configs/scaled_full_combined.yaml \
  --prepared data/prepared/europarl_full_combined \
  --tokenizer models/cpu_small/tokenizer \
  --out models/data_ablation/combined --device cuda

# 4. Compare on the same newstest2014 rows. Repeat for fr-en.
$PY -m nmt.cli evaluate --config configs/scaled_full_tatoeba.yaml \
  --tokenizer models/cpu_small/tokenizer \
  --benchmark data/prepared/newstest2014/newstest2014.en-fr.jsonl \
  --direction en-fr \
  --checkpoint baseline=models/data_ablation/baseline/best.pt \
  --checkpoint tatoeba=models/data_ablation/tatoeba/best.pt \
  --device cuda --bootstrap 1000 \
  --out reports/data_ablation.en-fr.json

# 5. Review-set plumbing check; automatic metrics only, references are not
#    human-reviewed. Repeat for fr-en.
$PY -m nmt.cli evaluate --config configs/scaled_full_tatoeba.yaml \
  --tokenizer models/cpu_small/tokenizer \
  --benchmark data/prepared/everyday_review/everyday_review.en-fr.jsonl \
  --direction en-fr \
  --checkpoint baseline=models/data_ablation/baseline/best.pt \
  --checkpoint tatoeba=models/data_ablation/tatoeba/best.pt \
  --device cuda --bootstrap 1000 \
  --out reports/everyday_review.en-fr.json
```

The four training arms are **not** automatically launched. Each is a full
350M-token run; the existing full-data run took 9h19m on the RTX 3050 Ti, so
four arms require roughly 36 GPU-hours plus evaluation. The current 800M-token
continuation must finish or be stopped deliberately before competing for the
same GPU.

## Validation actually performed

- `pytest -q`: **42 passed** on CPU, including the new source parsers, language
  and repetition filters, cross-source dedup, near-duplicate detection, review
  manifest, and a full mixture on the fixture baseline.
- Real-source smoke preparation: Tatoeba weekly files + an 8 MiB ParaCrawl
  prefix on the fixture baseline; 4,022 train / 59 dev / 65 test pairs, 0 exact
  cross-split overlaps, 0 near-duplicate train leaks, 0 review-set overlaps.
- Tokenizer trained on the smoke `train.tsv` only; 1,200-entry vocabulary,
  937 merges.
- Tiny CPU train: 10 updates, 9,842 target tokens, 2.33 s; this only exercises
  the data path. Its review output is not a quality result and is not reported
  as one.
- No full Tatoeba/ParaCrawl/combined model was trained, so no BLEU/chrF++ claim
  is made for those arms.

## Known limitations

- Tatoeba weekly exports do not expose per-row license or review metadata; the
  run uses the source-level CC BY 2.0 FR terms and retains IDs for attribution.
- ParaCrawl is sampled from a bounded prefix, not uniformly across the 21 GiB
  release. Domain and temporal bias are expected.
- ParaCrawl v9 TXT has no per-row quality score; heuristic filtering can miss
  semantic misalignments and can remove valid short/technical sentences.
- The near-duplicate check uses word 3-gram Jaccard, not semantic similarity, and
  can miss paraphrases.
- New source dev/test additions are small and do not replace the existing
  in-domain dev split or newstest2014.
- The everyday review set has no qualified bilingual human review. Its French
  references are drafts; treat any scores as plumbing checks only.
- Baseline exact/near-duplicate audits are reported but do not rewrite the
  frozen baseline; a pre-existing baseline issue would remain a documented
  limitation.
