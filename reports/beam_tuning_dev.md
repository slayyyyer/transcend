# Beam length-penalty and minimum-length tuning (dev only)

Checkpoint: step 14,500 (`models/accel_pilot/best.pt`, SHA-256
`cf813c77caed6e7e6785318bfa8630feb70d5f26ec2887f45735a2859c126231`).

Protocol: tune on the in-domain Europarl dev split only; newstest2014 is not used
for selection. The dev split has 5,321 rows per direction, one opaque block id per
row. To keep the sweep affordable, every 17th row was selected deterministically
(stride 17): 313 rows per direction, of which 305 EN→FR and 296 FR→EN remain after
the model's 128-source-token limit is applied. The same retained rows are used for
every configuration and for the greedy baseline.

| Direction | Configuration | BLEU | chrF++ | Mean words | ≤5 words | ≤8 words |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| EN→FR | greedy | 19.986 | 48.127 | 26.593 | 8 | 19 |
| EN→FR | beam 4, LP 1.0, min 0 | 22.033 | 48.904 | 25.059 | 9 | 18 |
| EN→FR | beam 4, LP 1.2, min 0 | 21.946 | **49.247** | 25.810 | 9 | 18 |
| EN→FR | beam 4, LP 1.2, min 8 | 21.946 | **49.247** | 25.810 | 9 | 18 |
| EN→FR | beam 4, LP 1.2, min 12 | 21.883 | 49.225 | 25.866 | 5 | 18 |
| EN→FR | beam 4, LP 1.2, min 16 | 21.750 | 49.203 | 25.990 | **0** | 13 |
| EN→FR | beam 4, LP 1.5, min 0 | 17.055 | 48.216 | 34.115 | 5 | 13 |
| FR→EN | greedy | 20.378 | 48.826 | 24.446 | 5 | 14 |
| FR→EN | beam 4, LP 1.0, min 0 | 23.325 | 49.593 | 22.693 | 6 | 19 |
| FR→EN | beam 4, LP 1.2, min 0 | 23.125 | **49.782** | 23.108 | 5 | 18 |
| FR→EN | beam 4, LP 1.2, min 8 | 23.122 | **49.784** | 23.111 | 5 | 18 |
| FR→EN | beam 4, LP 1.2, min 12 | 23.049 | 49.753 | 23.139 | 2 | 18 |
| FR→EN | beam 4, LP 1.2, min 16 | 22.838 | 49.704 | 23.345 | **0** | 12 |

## Selection

Selected: **beam 4, length penalty 1.2, minimum 16 new tokens**.

- LP 1.2 has the best dev chrF++ in both directions. LP 1.5 over-lengthens: mean
  output rises to 34 words EN→FR and BLEU falls from 21.95 to 17.06, so it is
  rejected.
- A minimum of 8 target tokens is a no-op on this sample. A minimum of 12 reduces
  ≤5-word outputs from 14 to 7 across the two directions, but still leaves short
  outputs.
- A minimum of 16 eliminates all ≤5-word outputs (14 → 0) and reduces ≤8-word
  outputs (33 → 25), while giving up 0.24 mean BLEU and 0.06 mean chrF++ relative
  to LP 1.2 with no minimum. This is a deliberate truncation-control trade-off,
  not a metric win.
- The selection rule was fixed before inspecting the full test: prefer LP 1.2,
  then the smallest minimum length that removes the observed ≤5-word output
  cluster without a material chrF++ loss. The dev sample is small (305/296), so
  the differences are noisy; the full-test result is reported separately and is
  not used to revise the setting.

Reproduce the sweep with:

```bash
PYTHONIOENCODING=utf-8 "$PY" scripts/tune_beam_dev.py \
  --config configs/existing_accelerator.yaml \
  --tokenizer models/cpu_small/tokenizer \
  --checkpoint models/accel_pilot/best.pt --device cuda \
  --directions en-fr,fr-en --grid '[[1.2,16]]' \
  --out reports/beam_tuning_dev/tuning_min16.json
```

Raw sweep outputs are in `reports/beam_tuning_dev/`. This is automatic metric
tuning on dev, not a human evaluation and not evidence of usable translation
quality.
