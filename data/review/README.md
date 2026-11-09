# Everyday review set

`everyday_review.tsv` is a small, project-authored diagnostic set for
constructions that the news/parliamentary training data under-represents:
everyday questions, negation, names, numbers/dates, idioms, and the
formal/informal *vous/tu* distinction.

Important status:

- **The French references have not received qualified bilingual human review.**
  Every row is labeled `not_bilingual_human_reviewed`.
- This is not a statistically representative benchmark and must not be reported
  as a general translation-quality estimate.
- The set is excluded from training. `prepare-mixture` checks it as held-out
  material; `nmt.cli prepare-review` writes the evaluation JSONL and manifest.
- The English prompts for the four *tu/vous* rows are intentionally similar so
  the two address forms can be compared.

To prepare it:

```bash
python -m nmt.cli prepare-review \
  --source data/review/everyday_review.tsv \
  --out data/prepared/everyday_review
```

The generated manifest records `human_reviewed: false`, category counts, source
and output hashes, and the exact limitation text.
