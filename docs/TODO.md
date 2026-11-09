# TODO — data expansion and next measured runs

Last updated: **2026-11-09**

This is the ordered backlog for turning the Tatoeba/ParaCrawl preparation work
into measured results. It is a plan, not a quality claim. Read
[`DATA_CARD.md`](DATA_CARD.md), [`DECISIONS.md`](DECISIONS.md), and
[`../reports/data_expansion_smoke.md`](../reports/data_expansion_smoke.md)
before changing the protocol.

## Current snapshot

| Item | State |
| --- | --- |
| Existing-data 350M baseline | Complete: step 40,113 / 350,004,440 tokens; dev-selected best is step 38,000 (`models/scaled_full/best.pre_extend.pt`). |
| 800M continuation | Complete: step 91,691 / 800,002,117 tokens; dev-selected best is step 77,000 (`models/scaled_full/best.pt`); final checkpoint is step 91,691. |
| 800M vs 350M comparison | Complete: all four greedy/beam-4, EN→FR/FR→EN cells improved; every paired chrF++ 95% interval excludes zero. See `reports/newstest2014.scaled_full_800m.comparisons.{json,md}`. |
| `models/scaled_full/best.pt` | Step 77,000 dev-selected best from the 800M continuation. **Do not use it as the 350M baseline; use `best.pre_extend.pt`.** |
| Tatoeba + ParaCrawl pipeline | Complete and smoke-tested only. No model has trained on either source. |
| Full Tatoeba/ParaCrawl/combined mixtures | Not built. |
| Three new 350M data arms | Not trained. |
| Everyday review set | 24 draft rows, 4 per category; not bilingual-human-reviewed. |
| GPU | Free; no training process is running. |

The four configs were checked at the start of this backlog: their model, seed,
tokenizer, sequence limits, optimizer, schedule, and 350M-token budget are
identical. The only meaningful difference is the `data.source` and config
`name`.

## Priority 0 — finish and evaluate the 800M continuation (complete)

- [x] The wrapper exited 0 after training and evaluation; no 24-hour walltime
      stop occurred.
- [x] `models/scaled_full/training_report.json` records step 91,691 /
      800,002,117 tokens, stop reason `max_train_tokens`, and best dev chrF++
      56.0833 at step 77,000.
- [x] The preserved artifacts exist: `best.pre_extend.pt` (step 38,000),
      `checkpoint.pre_extend.pt` (step 40,113), and the pre-extension report
      and history.
- [x] All four evaluations completed on the same retained newstest2014 rows:
      greedy and beam-4, EN→FR and FR→EN.
- [x] Inspected
      `reports/newstest2014.scaled_full_800m.comparisons.{json,md}`. The final
      step-91,691 checkpoint improves over the 350M baseline in all four cells;
      the smallest gain is +2.232 BLEU / +2.020 chrF++ (FR→EN beam 4). Every
      paired chrF++ 95% interval excludes zero. The 77,000 dev-selected best is
      also above baseline; the final test scores are slightly higher, but dev
      preferred 77,000 and final-vs-best was not bootstrap-tested.
- [x] `README.md`, `docs/MODEL_CARD.md`, `docs/DECISIONS.md`, and this backlog
      now record the measured result as automatic metrics only.

**Result:** 800M tokens produced a clear, consistent newstest2014 gain over the
350M checkpoint, but at roughly 2.3x the token budget. The next work is the
independent data-expansion arms, not another blind token increase.

## Priority 1 — build and audit the three full mixtures

These are CPU/disk jobs and the GPU is now free. They copy the baseline splits,
so expect roughly three additional full-corpus directories; check free space
before building all three.

- [ ] Rebuild the review manifest if it is missing:

      ```bash
      $PY -m nmt.cli prepare-review \
        --source data/review/everyday_review.tsv \
        --out data/prepared/everyday_review
      ```

- [ ] Build Tatoeba-only on top of the existing full baseline:

      ```bash
      $PY -m nmt.cli prepare-mixture \
        --base data/prepared/europarl_full \
        --out data/prepared/europarl_full_tatoeba \
        --include-tatoeba --tatoeba-max-pairs 100000 \
        --review-dir data/prepared/everyday_review
      ```

- [ ] Build ParaCrawl-only using the bounded 64 MiB prefix:

      ```bash
      $PY -m nmt.cli prepare-mixture \
        --base data/prepared/europarl_full \
        --out data/prepared/europarl_full_paracrawl \
        --include-paracrawl --paracrawl-max-pairs 100000 \
        --paracrawl-max-bytes 67108864 \
        --review-dir data/prepared/everyday_review
      ```

- [ ] Build the combined mixture:

      ```bash
      $PY -m nmt.cli prepare-mixture \
        --base data/prepared/europarl_full \
        --out data/prepared/europarl_full_combined \
        --include-tatoeba --include-paracrawl \
        --tatoeba-max-pairs 100000 --paracrawl-max-pairs 100000 \
        --paracrawl-max-bytes 67108864 \
        --review-dir data/prepared/everyday_review
      ```

- [ ] For each output, inspect `manifest.json` and
      `contamination_report.json`. Record:
      - actual selected pairs per source (a cap is a ceiling, not a quota);
      - filtering drops and the retained ParaCrawl prefix size/hash;
      - train/dev/test counts and source split;
      - exact cross-split overlaps (must be 0);
      - near-duplicate train leaks and new-held-out overlaps;
      - review-set overlaps (must be 0);
      - whether the baseline rows were copied verbatim.
- [ ] Do not silently rebuild or modify `data/prepared/europarl_full`; it is the
      frozen baseline for this comparison.
- [ ] Record the full-preparation counts in `docs/DATA_CARD.md` or a new report.
      Do not copy the smoke statistics as if they were full-corpus results.

**Done when:** all three manifests exist, their assertions pass, and the actual
counts are written down.

## Priority 2 — train and compare the three new 350M arms

The existing 350M baseline is already the correct from-scratch run. Reuse it
rather than spending another ~9 GPU-hours:

- Baseline checkpoint: `models/scaled_full/best.pre_extend.pt` (step 38,000).
- Baseline evaluation already exists under
  `reports/newstest2014.scaled_full.*`, but re-evaluate the baseline alongside
  the new arms so every checkpoint uses exactly the same retained rows and
  decoding settings.

- [ ] Confirm the frozen tokenizer is still
      `models/cpu_small/tokenizer` and do not retrain or retokenize it.
- [ ] Train Tatoeba-only:

      ```bash
      $PY -m nmt.cli train --config configs/scaled_full_tatoeba.yaml \
        --prepared data/prepared/europarl_full_tatoeba \
        --tokenizer models/cpu_small/tokenizer \
        --out models/data_ablation/tatoeba --device cuda
      ```

- [ ] Train ParaCrawl-only:

      ```bash
      $PY -m nmt.cli train --config configs/scaled_full_paracrawl.yaml \
        --prepared data/prepared/europarl_full_paracrawl \
        --tokenizer models/cpu_small/tokenizer \
        --out models/data_ablation/paracrawl --device cuda
      ```

- [ ] Train combined:

      ```bash
      $PY -m nmt.cli train --config configs/scaled_full_combined.yaml \
        --prepared data/prepared/europarl_full_combined \
        --tokenizer models/cpu_small/tokenizer \
        --out models/data_ablation/combined --device cuda
      ```

- [ ] Run these sequentially, not concurrently on the 4 GB GPU. At the measured
      full-data rate, each is about 9h19m; budget roughly 28 GPU-hours for the
      three new arms plus evaluation.
- [ ] After each run, verify the stop reason and token count. If a run stops on
      walltime before 350M tokens, resume its checkpoint; do not change the
      budget or batch size mid-arm.
- [ ] Evaluate all four checkpoints on exactly the same newstest2014 rows:
      baseline, Tatoeba, ParaCrawl, combined. Run greedy and beam-4 (length
      penalty 1.2, minimum 16 new tokens) in both directions, with 1,000
      document-bootstrap resamples.
- [ ] Write one comparison report containing all eight cells per arm
      (2 directions × 2 decoders), paired 95% intervals, and sentence-level
      W/L/T. Do not select the aggregate metric after seeing the test result.
- [ ] Treat an arm as a candidate only if its gains are consistent across both
      directions and both decoders. Report mixed or negative results as mixed
      or negative. A 100k-pair addition is only a small fraction of the 2.1M-pair
      baseline, so a neutral result is a real possible outcome.

**Done when:** the four-way report exists and the next action is justified by
the measured comparison, not by the smoke run.

## Priority 3 — make the everyday review set defensible

The current 24 rows are a plumbing check, not a benchmark. All references are
`not_bilingual_human_reviewed`.

- [ ] Decide the target size; recommendation: at least 100 rows per category
      (questions, negation, names, numbers/dates, idioms, *tu/vous*), or at
      least 300 total if time is constrained.
- [ ] Have a qualified bilingual reviewer check every reference. Record
      `reference_review=bilingual_human_reviewed` only after that happens.
- [ ] Keep the set completely outside train/dev/test and outside model
      selection.
- [ ] Rebuild `data/prepared/everyday_review` after review and run the review
      evaluation only as a diagnostic. Keep it separate from newstest2014 and do
      not merge its rows into the benchmark.
- [ ] If no qualified reviewer is available, keep the status as unreviewed and
      say so in every report; do not call the scores a human evaluation.

**Done when:** either the set is human-reviewed and labeled, or the unreviewed
status is explicit in all results and documentation.

## Priority 4 — resolve licensing before publishing an expanded model

- [ ] Decide whether Tatoeba's CC BY 2.0 FR attribution is compatible with the
      intended distribution. If Tatoeba is used, carry the attribution and
      source IDs into `NOTICE`, the data card, and the model card.
- [ ] Do not assume the weekly sentence files provide a reliable per-row
      license filter; they do not. If permissive redistribution is required and
      the attribution terms are unacceptable, drop Tatoeba rather than guessing
      which rows are CC0.
- [ ] Keep ParaCrawl's distinction explicit: the packaging is CC0, but the
      underlying website text is not owned or cleared by ParaCrawl. Do not claim
      that a ParaCrawl-trained artifact is freely redistributable.
- [ ] Re-run the artifact/provenance update after training: record the exact
      mixture manifest, source hashes, filtering version, and tokenizer hash.
      The current model cards correctly describe preparation-only sources; they
      must be updated if a source is actually trained on.

**Done when:** the model card and NOTICE state exactly which sources are in the
reported checkpoint and what redistribution limits apply.

## Priority 5 — decide what to scale

- [ ] If one new mixture wins consistently at 350M, train a fresh 800M model on
      that exact mixture with the same architecture/tokenizer/budget and compare
      it with the 800M baseline. Do not compare a 350M arm with an 800M arm.
- [ ] If no arm wins, keep the existing baseline and document the negative
      result. Do not add OpenSubtitles/GlobalVoices/WikiMatrix/Giga-French as a
      consolation experiment.
- [x] The 800M baseline beat the preserved 350M baseline in all four measured
      cells; record that as a positive but diminishing-return result. Do not
      assume further token increases will help.
- [ ] Decide whether any data-expansion arm justifies a fresh 800M run only
      after its 350M comparison. Do not extrapolate from the arm's training
      loss or the smoke test.
- [ ] Export and verify a chosen checkpoint only after the model comparison is
      complete. Existing checkpoints must not be retokenized.

## Explicitly out of scope for this phase

- Full 21 GiB ParaCrawl download or uniform sampling across the release.
- OpenSubtitles, GlobalVoices, WikiMatrix, Giga-French, DGT, or WMT News
  Commentary v18.1.
- New architecture, new tokenizer, or changing the 8,000-entry vocabulary.
- Human-quality claims from the 24-row review set or from smoke runs.
- Publishing an expanded model before the licensing review is complete.

## Completion criteria for this phase

- [x] 800M continuation evaluated against the preserved 350M checkpoint.
- [ ] Three full mixtures prepared with audited manifests and zero overlap
      assertion failures.
- [ ] Three new 350M arms trained with the frozen tokenizer.
- [ ] Four-way newstest2014 comparison reported for greedy and beam-4 in both
      directions, with paired intervals and sentence-level results.
- [ ] If a data arm wins, train it at 800M and compare against the 800M
      baseline.
- [ ] Review-set status is honest and separate from the benchmark.
- [ ] Licensing and provenance are documented for every source actually used.
- [ ] No quality improvement is claimed unless the measured comparison supports
      it.
