# Full data-expansion preparation report

Date: 2026-11-09. This report records **preparation and auditing only**. No
model had been trained on Tatoeba, ParaCrawl, or the combined mixture when this
audit was written; the Tatoeba-only 350M arm was launched afterward. This
report contains no training result, and no translation-quality improvement is
claimed.

## Inputs

- Baseline: frozen `data/prepared/europarl_full`, 2,101,321 train / 20,138 dev /
  19,789 test pairs. Its split and manifest SHA-256 values were recorded before
  the runs and checked again afterward.
- Tatoeba weekly export 2026-11-07: English and French sentence files plus the
  EN–FR link file.
- ParaCrawl v9 EN–FR TXT: bounded 64 MiB compressed prefix, HTTP Range request
  against the 21,863,614,702-byte object. Prefix SHA-256:
  `360d51948a00831e21b5b7937a2774d44516f04ea9531a6f2bc4584d8d2a444a`.
- Review set: `data/prepared/everyday_review`, 24 project-authored draft rows
  (4 per category), still `not_bilingual_human_reviewed`.

The prior review draft `i02` (“Break a leg!” / “Bonne chance !”) was found to be
an exact baseline-dev row. It was replaced with “Once in a blue moon.” / “Tous
les trente-six du mois.” and the review manifest was rebuilt. The current review
set has zero exact overlap with the frozen baseline.

## Commands

The three runs used the documented `prepare-mixture` command with these fixed
settings:

```bash
--base data/prepared/europarl_full
--review-dir data/prepared/everyday_review
--skip-base-near-audit
```

Tatoeba and ParaCrawl each used a 100,000-pair cap. ParaCrawl used
`--paracrawl-max-bytes 67108864`. The combined run used both sources and both
caps. `--skip-base-near-audit` skips only the optional audit of the **frozen
baseline**; it does not skip the new-source near-duplicate checks.

## Results

| Arm | Added train | Added dev | Added test | Added total | Final train | Final dev | Final test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tatoeba | 97,064 | 1,007 | 1,064 | 99,135 | 2,198,385 | 21,145 | 20,853 |
| ParaCrawl | 98,034 | 812 | 1,073 | 99,919 | 2,199,355 | 20,950 | 20,862 |
| Combined | 195,094 | 1,819 | 2,137 | 199,050 | 2,296,415 | 21,957 | 21,926 |

The combined run selected 199,630 new pairs after new-source deduplication
(370 removed: 366 Tatoeba, 4 ParaCrawl), then removed 473 exact base/held-out
overlaps and 107 near-duplicates. The 100,000 caps are ceilings, not guaranteed
additions.

### Source cleaning

| Source | Input pairs | Kept after cleaning | Selected | Main drops |
| --- | ---: | ---: | ---: | --- |
| Tatoeba | 442,283 | 440,351 | 100,000 | 1,691 repetition, 139 ratio, 81 length, 19 identical, 2 malformed |
| ParaCrawl | 665,356 | 489,031 | 100,000 | 99,768 repetition, 76,171 language, 382 malformed, 4 length |

Tatoeba removed 26 cleaned pairs before sampling because they exactly matched
the review set. ParaCrawl removed zero review rows. The combined run read
1,107,639 source pairs and retained 929,382 after cleaning.

### Contamination and deduplication

All nine final assertions are **0** for every arm:

- train/dev, train/test, and dev/test pair-id overlaps;
- train/dev, train/test, and dev/test English-text overlaps;
- train/dev, train/test, and dev/test French-text overlaps.

New-source near-duplicate checks at Jaccard 0.85:

| Arm | New train removed vs held-out | New held-out removed vs base train | New dev/test removed vs review |
| --- | ---: | ---: | ---: |
| Tatoeba | 48 | 22 | 0 |
| ParaCrawl | 37 | 0 | 0 |
| Combined | 85 | 22 | 0 |

Exact one-side filtering was added because near-duplicate shingles can miss
very common short sentences. For the combined arm the exact filters removed
162 new-train duplicates of base train, 40 train rows sharing a side with base
held-out text, 247 train rows sharing a side with new held-out text, 2+13 new
dev rows and 1+8 new test rows overlapping base train. No review row remains in
any final split.

## Baseline preservation

For every arm, the first 2,101,321 train rows, 20,138 dev rows, and 19,789 test
rows hash to the original baseline files:

- train `e971acfe29f79806ce09c97e4cd7c59b0ec8d7c4b271183c84a8963b9ff46887`
- dev `1265139bca341cd35eca30a4024d212dd62fe4d6b790df43aae32a1d7923b73c`
- test `c7f3b61069f38b4f81a265d749ce6b97a679ae0b9a788a399e0d7e335df86a00`

`sha256sum -c` on the baseline train/dev/test and manifest passed after all
runs. Each output’s actual split hashes also match its manifest. The output
directories are 715 MB (Tatoeba), 727 MB (ParaCrawl), and 736 MB (combined).

## Manifest locations

- `data/prepared/europarl_full_tatoeba/{manifest.json,contamination_report.json}`
- `data/prepared/europarl_full_paracrawl/{manifest.json,contamination_report.json}`
- `data/prepared/europarl_full_combined/{manifest.json,contamination_report.json}`

Every manifest records the source acquisition metadata, cleaning/sampling
counts, exact and near-duplicate results, review status, split hashes, and the
frozen tokenizer policy.

## Limitations

- The optional **baseline-wide** near-duplicate audit was skipped for runtime
  (`audit_base_near_duplicates: false`). This is not reported as a passed check.
  The baseline was also not modified; its exact cross-split assertions remain 0.
- Tatoeba is capped by English-sentence group, but 100,000 is a ceiling. The
  final additions are smaller after base/held-out filtering.
- ParaCrawl is a deterministic hash cap over a bounded 64 MiB compressed
  prefix, not a uniform sample of the 21 GiB release. The TXT has no per-row
  quality score; heuristic filtering can miss semantic misalignments.
- The review set has no qualified bilingual human review. It is a diagnostic
  set, not a benchmark.
- No completed model exists for these mixtures. The Tatoeba-only 350M arm was
  launched after this audit; training and evaluation remain Priority 2.

## Verification run

- `pytest -q`: **43 passed** on CPU.
- New tests cover review-row removal before sampling and the mixture/review
  manifest path.
- All three preparation commands exited 0; no split assertion failed.
- All output hashes match their manifests; all baseline prefix hashes match the
  frozen baseline.
