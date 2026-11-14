# TODO — data expansion and next measured runs

Last updated: **2026-11-14**

This is the ordered backlog for turning the Tatoeba/ParaCrawl preparation work
into measured results. It is a plan, not a quality claim. Read
[`DATA_CARD.md`](DATA_CARD.md), [`DECISIONS.md`](DECISIONS.md),
[`../reports/data_expansion_smoke.md`](../reports/data_expansion_smoke.md), and
[`../reports/data_expansion_full.md`](../reports/data_expansion_full.md)
before changing the protocol.

## Current snapshot

| Item | State |
| --- | --- |
| Existing-data 350M baseline | Complete: step 40,113 / 350,004,440 tokens; dev-selected best is step 38,000 (`models/scaled_full/best.pre_extend.pt`). |
| 800M continuation | Complete: step 91,691 / 800,002,117 tokens; dev-selected best is step 77,000 (`models/scaled_full/best.pt`); final checkpoint is step 91,691. |
| 800M vs 350M comparison | Complete: all four greedy/beam-4, EN→FR/FR→EN cells improved; every paired chrF++ 95% interval excludes zero. See `reports/newstest2014.scaled_full_800m.comparisons.{json,md}`. |
| `models/scaled_full/best.pt` | Step 77,000 dev-selected best from the 800M continuation. **Do not use it as the 350M baseline; use `best.pre_extend.pt`.** |
| Tatoeba + ParaCrawl pipeline | Complete; full mixtures built and audited. Tatoeba and ParaCrawl arms completed and evaluated. |
| Full Tatoeba/ParaCrawl/combined mixtures | Built and audited. See `reports/data_expansion_full.md`. |
| Three new 350M data arms | Tatoeba and ParaCrawl complete and evaluated at 19.8M/350M; the 350M combined arm is still unlaunched. |
| Everyday review set | 24 draft rows, 4 per category; baseline-clean, not bilingual-human-reviewed. |
| 800M ParaCrawl | Complete: step 91,754 / 800,004,736 tokens, stop `max_train_tokens`; dev-selected best step 91,000 at 56.4130 mean dev chrF++; all four newstest2014 cells beat the 800M step-77,000 baseline. See `reports/newstest2014.paracrawl_800m.comparisons.{json,md}`. |
| 50M scale-check baseline | Complete: step 41,373 / 350,001,072 tokens, stop `max_train_tokens`; dev-selected best step 41,000 at 56.2123 mean dev chrF++; 48,236,544 parameters. No newstest2014 evaluation yet. |
| 50M scale-check combined | Training: started 2026-11-14 07:47 EST from random initialization; output `models/scale50/combined`; log `reports/scale50_combined.log`. No checkpoint or evaluation result yet. |
| GPU | Busy: 50M combined arm (Python PID 10756). |

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

## Priority 1 — build and audit the three full mixtures (complete)

These are CPU/disk jobs and the GPU is now free. They copy the baseline splits,
so expect roughly three additional full-corpus directories; check free space
before building all three.

- [x] Rebuild the review manifest if it is missing (24 diagnostic rows; the
      optional 200-row gate is deferred):

      ```bash
      $PY -m nmt.cli prepare-review \
        --source data/review/everyday_review.tsv \
        --out data/prepared/everyday_review
      ```

- [x] Build Tatoeba-only on top of the existing full baseline:

      ```bash
      $PY -m nmt.cli prepare-mixture \
        --base data/prepared/europarl_full \
        --out data/prepared/europarl_full_tatoeba \
        --include-tatoeba --tatoeba-max-pairs 100000 \
        --review-dir data/prepared/everyday_review
      ```

- [x] Build ParaCrawl-only using the bounded 64 MiB prefix:

      ```bash
      $PY -m nmt.cli prepare-mixture \
        --base data/prepared/europarl_full \
        --out data/prepared/europarl_full_paracrawl \
        --include-paracrawl --paracrawl-max-pairs 100000 \
        --paracrawl-max-bytes 67108864 \
        --review-dir data/prepared/everyday_review
      ```

- [x] Build the combined mixture:

      ```bash
      $PY -m nmt.cli prepare-mixture \
        --base data/prepared/europarl_full \
        --out data/prepared/europarl_full_combined \
        --include-tatoeba --include-paracrawl \
        --tatoeba-max-pairs 100000 --paracrawl-max-pairs 100000 \
        --paracrawl-max-bytes 67108864 \
        --review-dir data/prepared/everyday_review
      ```

- [x] For each output, inspect `manifest.json` and
      `contamination_report.json`. Record:
      - actual selected pairs per source (a cap is a ceiling, not a quota);
      - filtering drops and the retained ParaCrawl prefix size/hash;
      - train/dev/test counts and source split;
      - exact cross-split overlaps (must be 0);
      - near-duplicate train leaks and new-held-out overlaps;
      - review-set overlaps (must be 0);
      - whether the baseline rows were copied verbatim.
- [x] Do not silently rebuild or modify `data/prepared/europarl_full`; it is the
      frozen baseline for this comparison.
- [x] Record the full-preparation counts in `docs/DATA_CARD.md` and
      `reports/data_expansion_full.md`. Do not copy the smoke statistics as if
      they were full-corpus results.

**Done when:** all three manifests exist, their assertions pass, and the actual
counts are written down.

**Result (2026-11-09):** Tatoeba 2,198,385 / 21,145 / 20,853 (added 99,135);
ParaCrawl 2,199,355 / 20,950 / 20,862 (added 99,919); combined
2,296,415 / 21,957 / 21,926 (added 199,050). All nine assertions are zero in
each arm and the baseline prefixes are byte-identical. The optional
baseline-wide near audit was skipped for runtime and is not a passed check.
Full audit: [`../reports/data_expansion_full.md`](../reports/data_expansion_full.md).

## Priority 2 — train and compare the three new 350M arms

The existing 350M baseline is already the correct from-scratch run. Reuse it
rather than spending another ~9 GPU-hours:

- Baseline checkpoint: `models/scaled_full/best.pre_extend.pt` (step 38,000).
- Baseline evaluation already exists under
  `reports/newstest2014.scaled_full.*`, but re-evaluate the baseline alongside
  the new arms so every checkpoint uses exactly the same retained rows and
  decoding settings.

- [x] Confirm the frozen tokenizer is still
      `models/cpu_small/tokenizer` and do not retrain or retokenize it. The
      merge hash matches the baseline: `1b8eda9c55e92120056b2bb213a09899e677d488b1f444681ad508d6b838541e`.
- [x] Train Tatoeba-only (completed 2026-11-09):

      ```bash
      $PY -m nmt.cli train --config configs/scaled_full_tatoeba.yaml \
        --prepared data/prepared/europarl_full_tatoeba \
        --tokenizer models/cpu_small/tokenizer \
        --out models/data_ablation/tatoeba --device cuda
      ```

      Result: completed at step 40,149 / 350,000,282 tokens; dev-selected best
      is step 39,000. On newstest2014, FR→EN gains were +0.543/+0.336 BLEU and
      +0.423/+0.351 chrF++ (greedy/beam 4), with the chrF++ 95% intervals
      excluding zero; EN→FR was positive but not significant. No cell
      regressed. See `reports/newstest2014.tatoeba.comparisons.{json,md}`.
      Automatic metrics only.

- [x] Train ParaCrawl-only (completed 2026-11-11):

      ```bash
      $PY -m nmt.cli train --config configs/scaled_full_paracrawl.yaml \
        --prepared data/prepared/europarl_full_paracrawl \
        --tokenizer models/cpu_small/tokenizer \
        --out models/data_ablation/paracrawl --device cuda
      ```

      Result: finished at step 40,144 / 350,001,418 tokens on
      `max_train_tokens`; dev-selected best step 40,000. On newstest2014, all
      four cells improved: +1.313/+1.228 BLEU and +1.017/+0.922 chrF++ greedy,
      and +1.141/+1.136 BLEU and +0.882/+0.888 chrF++ with beam 4 (EN→FR/FR→EN).
      Every paired chrF++ 95% interval excludes zero. See
      `reports/newstest2014.paracrawl.comparisons.{json,md}`. Automatic metrics
      only.

- [ ] Train combined:

      ```bash
      $PY -m nmt.cli train --config configs/scaled_full_combined.yaml \
        --prepared data/prepared/europarl_full_combined \
        --tokenizer models/cpu_small/tokenizer \
        --out models/data_ablation/combined --device cuda
      ```

- [ ] Run the combined arm sequentially, not concurrently on the 4 GB GPU. The
      remaining arm requires roughly 9 GPU-hours plus evaluation.
- [x] After the Tatoeba and ParaCrawl runs, verified both stopped on
      `max_train_tokens` at the full 350M-token budget. Apply the same check to
      combined; if it stops on walltime before 350M tokens, resume its
      checkpoint rather than changing the budget or batch size.
- [x] Evaluate the baseline, Tatoeba, and ParaCrawl checkpoints on exactly the
      same newstest2014 rows. Ran greedy and beam-4 (length penalty 1.2,
      minimum 16 new tokens) in both directions, with 1,000 document-bootstrap
      resamples.
- [ ] Evaluate combined on the same rows and decoding settings.
- [x] Write separate four-cell comparison reports for Tatoeba and ParaCrawl,
      each with paired 95% intervals and sentence-level W/L/T.
- [ ] Add combined and write the final four-way report. Do not select the
      aggregate metric after seeing the test result.
- [x] Applied the consistency rule: Tatoeba is reported as direction-specific;
      ParaCrawl is positive in both directions and both decoders at 350M.
      Report mixed or negative results as mixed or negative. A 100k-pair
      addition is only a small fraction of the 2.1M-pair baseline, so a neutral
      result is a real possible outcome.

**Done when:** the four-way report exists and the next action is justified by
the measured comparison, not by the smoke run.

## Priority 2b — 50M-parameter scale check

This is a separate model-size axis, not another 350M-token data arm. The 19.8M
model is capacity-limited on the 4 GB card, so a 48.2M-parameter model is being
checked with the same data recipes. Do not mix these results into the 19.8M
four-way comparison.

- [x] Wrote the matched config pair: `configs/scale50_full.yaml` (full Europarl
      v7 + News Commentary v9) and `configs/scale50_combined.yaml` (adds the
      audited Tatoeba and bounded ParaCrawl mixture). Both use a
      48,236,544-parameter 6+6 model (`d_model` 512, 8 heads, FFN 2,048), seed
      1337, the frozen 8,000-entry tokenizer, 128/136 sequence limits, peak LR
      3e-4, 1,000 warmup updates, an 8,192-token update budget and a 350M-token /
      45,000-step / 40-hour cap. Only `data.source` and the name differ.
- [x] Baseline completed (2026-11-14): step 41,373 / 350,001,072 tokens, stop
      `max_train_tokens`, 109,110.43 s cumulative walltime, 5,719.76 target
      tokens/s, 1,787.78 ms mean optimizer update. Dev-selected best is step
      41,000 at 56.2123 mean chrF++; `best.pt` is that checkpoint and
      `checkpoint.pt` is step 41,373.
- [ ] Combined arm training (started 2026-11-14 07:47 EST). Do not start a
      second training process; the GPU is saturated by this run.
- [ ] When it stops, evaluate both 50M checkpoints on exactly the same retained
      newstest2014 rows with greedy and beam 4 (length penalty 1.2, minimum 16
      new tokens), and write
      `reports/newstest2014.scale50_combined.comparisons.{json,md}`.
- [ ] Report the result as automatic metrics only. A 50M model trained on these
      corpora is still not a production translator, whatever the BLEU.

**Done when:** the combined arm stops on `max_train_tokens` and the four-cell
comparison exists, or the experiment is documented as stopped early with the
reason.

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
      The current model cards describe the Europarl-only models; they must be
      updated before any trained expansion arm is released.

**Done when:** the model card and NOTICE state exactly which sources are in the
reported checkpoint and what redistribution limits apply.

## Priority 5 — decide what to scale

- [x] ParaCrawl met the consistency bar at 350M, so it was trained fresh at
      800M with the same architecture/tokenizer/budget (`configs/scaled_full_paracrawl_800m.yaml`,
      `models/data_ablation/paracrawl_800m`) and compared with the 800M baseline,
      not with a 350M checkpoint. Result: step 91,754 / 800,004,736 tokens, stop
      `max_train_tokens`; dev-selected best step 91,000 at 56.4130 mean chrF++.
      Against `models/scaled_full/best.pt` (step 77,000), all four newstest2014
      cells improved: +1.153/+1.079 BLEU/chrF++ EN→FR greedy and +1.432/+1.011
      beam 4; +1.476/+1.085 FR→EN greedy and +1.560/+1.103 beam 4. Every paired
      chrF++ 95% interval excludes zero. See
      `reports/newstest2014.paracrawl_800m.comparisons.{json,md}`. Automatic
      metrics only.
- [ ] If no arm wins, keep the existing baseline and document the negative
      result. Do not add OpenSubtitles/GlobalVoices/WikiMatrix/Giga-French as a
      consolation experiment.
- [x] The 800M baseline beat the preserved 350M baseline in all four measured
      cells; record that as a positive but diminishing-return result. Do not
      assume further token increases will help.
- [x] Decided ParaCrawl qualifies for a fresh 800M run after its 350M
      comparison; Tatoeba does not meet the all-four-cell consistency bar on its
      own. Do not extrapolate from training loss or the smoke test.
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
- [x] Three full mixtures prepared with audited manifests and zero overlap
      assertion failures.
- [x] Tatoeba and ParaCrawl 350M arms trained and evaluated with the frozen
      tokenizer.
- [ ] Combined 350M arm trained with the frozen tokenizer.
- [ ] Four-way newstest2014 comparison reported for greedy and beam-4 in both
      directions, with paired intervals and sentence-level results.
- [x] The 350M winner (ParaCrawl) trained at 800M and beat the 800M baseline in
      all four cells (800M vs 800M, not 350M vs 800M).
- [x] 50M scale-check baseline trained from random initialization to the full
      350M-token budget (`models/scale50/full`, step 41,373).
- [ ] 50M combined arm trained and compared with that baseline. Training is
      running; no combined result exists yet.
- [ ] Review-set status is honest and separate from the benchmark.
- [ ] Licensing and provenance are documented for every source actually used.
- [ ] No quality improvement is claimed unless the measured comparison supports
      it.
