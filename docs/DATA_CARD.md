# Data card

## Sources

| Source | Owner | Release | URL | Size | Terms |
| --- | --- | --- | --- | --- | --- |
| Europarl | European Parliament (distributed by statmt.org) | v7 | https://www.statmt.org/europarl/v7/fr-en.tgz | 202,718,517 B | Research distribution; no standard permissive licence; OPUS lists licence unknown |
| News Commentary | WMT / article publishers | v9 (WMT14) | https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz | 80,418,416 B | WMT14 task data; redistribution/commercial rights not granted by WMT |
| newstest2014 | WMT | WMT14 `test-full` | https://www.statmt.org/wmt14/test-full.tgz | 3,255,445 B | Evaluation set; used only for scoring |
| newstest2013 | WMT | WMT13 `test` | https://www.statmt.org/wmt13/test.tgz | 959,311 B | Recorded for provenance only; archive has sources, no EN/FR reference pair |

SHA-256 of every archive is recorded in `data/prepared/*/manifest.json`.

## Counts after cleaning

| Stage | Pairs |
| --- | --- |
| Raw sampled pairs (Europarl stride 5 + NC stride 1) | 584,796 |
| Empty after normalization | 1,475 |
| English == French | 1,319 |
| Exact duplicates | 5,364 |
| Length-ratio rejects (> 3.0 or < 1/3) | 547 |
| Kept | 576,091 |
| Train / dev / test | 565,475 / 5,321 / 5,098 |

Cleaning keeps accents and case, applies NFC, collapses ASCII whitespace, drops
markup-looking rows (none found) and removes exact duplicates. Rejection reasons
are counted in the manifest.

## Splitting and leakage

* Split unit: opaque contiguous block of 200 aligned pairs, hashed with a fixed
  salt into train/dev/test (98/1/1).
* Europarl and News Commentary are flat line-aligned releases with no document
  ids. The block is a locality device, **not** a document id, and is named as
  such everywhere.
* Any train pair whose English or French side appears in dev or test is removed.
  Counts are in `contamination_report.json`; the preparation step raises if any
  overlap remains.
* Both translation directions are derived from a pair only after its split is
  fixed, so directions cannot straddle splits.

## Known limitations

* Parliamentary and news text: no conversational or general-domain guarantee.
* Europarl v7 covers 1996–2011 sessions; vocabulary and topics are dated.
* No document-level context; each segment is translated independently.
* The corpus is not redistributed by this repository. Raw archives are
  git-ignored and must be re-fetched with `nmt prepare`.
* Derived model weights inherit the unresolved status of the training data and
  are published as research artifacts only.
