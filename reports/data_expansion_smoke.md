# Data-expansion smoke report

Date: 2026-11-09 (UTC). This report records preparation and plumbing validation
only. No translation-quality improvement is claimed, and no full ablation run
was launched.

## What ran

- Real Tatoeba weekly files from 2026-11-07: English and French sentences plus
  EN–FR links.
- A real bounded prefix of ParaCrawl v9 EN–FR TXT (8 MiB compressed for this
  smoke; the default is 64 MiB).
- The project fixture baseline, so the full mixture path could run cheaply
  without rebuilding or altering `data/prepared/europarl_full`.
- Review set prepared with `human_reviewed: false`.
- A 1,200-entry tokenizer trained on the smoke `train.tsv` only.
- A 10-update CPU train and a one-row review evaluation to exercise the
  evaluation path.

## Commands

```bash
python -m nmt.cli prepare-review \
  --source data/review/everyday_review.tsv \
  --out data/prepared/everyday_review

python -m nmt.cli prepare-mixture \
  --base data/prepared/fixtures \
  --out data/prepared/smoke_everyday \
  --include-tatoeba --include-paracrawl \
  --tatoeba-max-pairs 2000 --paracrawl-max-pairs 2000 \
  --paracrawl-max-bytes 8388608 \
  --review-dir data/prepared/everyday_review \
  --near-duplicate-threshold 0.85

python -m nmt.cli tokenizer \
  --train-dir data/prepared/smoke_everyday \
  --out models/smoke_everyday/tokenizer \
  --vocab-size 1200 --min-pair-frequency 1 \
  --max-source-tokens 64 --max-target-tokens 72

python -m nmt.cli train \
  --config configs/smoke.yaml \
  --prepared data/prepared/smoke_everyday \
  --tokenizer models/smoke_everyday/tokenizer \
  --out models/smoke_everyday/run --device cpu \
  --max-steps 10 --max-train-tokens 50000 --max-walltime 120

python -m nmt.cli evaluate \
  --config configs/smoke.yaml \
  --tokenizer models/smoke_everyday/tokenizer \
  --benchmark data/prepared/everyday_review/everyday_review.en-fr.jsonl \
  --direction en-fr \
  --checkpoint smoke=models/smoke_everyday/run/checkpoint.pt \
  --out reports/smoke_everyday_review.json --device cpu --limit 1 --bootstrap 10
```

The preparation command took about 64 seconds in this environment; tokenizer +
10-update train took about 11 seconds. Both are CPU/small-sample measurements,
not full-corpus estimates.

## Source results

The smoke used caps of 2,000 pairs per source after cleaning and a deterministic
hash cap. The acquisition cache was reused for Tatoeba; the ParaCrawl prefix was
downloaded with an HTTP Range request.

| Source | Raw/read pairs | Kept after cleaning | Selected | Split result (train/dev/test) |
| --- | ---: | ---: | ---: | --- |
| Tatoeba | 442,283 | 440,351 | 2,000 | 1,954 / 21 / 21 |
| ParaCrawl | 83,151 | 61,152 | 2,000 | 1,944 / 25 / 31 |

Tatoeba read 442,791 links, with 508 links missing a sentence on at least one
side. ParaCrawl's 8 MiB prefix contained 83,151 two-column rows and no score
column. The cleaned totals across both sources were 501,503; 3,996 new pairs
remained after cross-source deduplication and caps.

Final smoke corpus: **4,022 train / 59 dev / 65 test** (including the 124 / 13 /
13 fixture baseline copies).

## Filtering and leakage checks

- Exact cross-split assertions: all nine checked overlap counts were **0**.
- Near-duplicate train-vs-held-out: **0 removed** at 0.85.
- Near-duplicate new-dev/new-test vs base train: **0 removed**.
- Review-set exact overlap: **0**; review excluded from training.
- Base-train near-duplicate audit: **0 removed** on the fixture baseline.
- No source pair was silently truncated. ParaCrawl's bounded prefix is marked
  partial; the incomplete trailing gzip row is discarded.

The smoke shows the guards are active; it does not demonstrate that a full
2.1M-pair baseline plus 100k-pair source additions will be free of duplicates
or near-duplicates.

## Tokenizer and model plumbing

The smoke tokenizer had 1,200 entries / 937 merges and was trained from the
smoke `train.tsv` only. The 10-update CPU run processed 9,842 target tokens in
2.33 s (1,079,808 parameters) and wrote a resumable checkpoint. The one-row
review evaluation returned an empty hypothesis, which is expected from a
10-update model and is **not** a translation-quality measurement.

## Checksums and source metadata

The acquisition manifest records full SHA-256 values for the three Tatoeba
weekly files. The ParaCrawl prefix SHA-256 is specific to the 8 MiB smoke sample;
the full object is 21,863,614,702 bytes with ETag
`"a3838d27714582f1752b1b6a3640caea-2607"`. The default 64 MiB sample will have a
different prefix hash, recorded in the run manifest.

## Not run

- No full Tatoeba-only, ParaCrawl-only, or combined 350M-token training.
- No comparison of BLEU/chrF++ between data arms.
- No qualified bilingual review of the new everyday references.
- No uniform sample across the 21 GiB ParaCrawl release; the current sample is a
  bounded prefix and is documented as such.

The exact controlled commands and resource expectations are in
`docs/DATA_CARD.md`. Do not launch the four arms while the existing 800M-token
continuation owns the GPU.
