# Local artifacts

Large corpora and trained artifacts are intentionally **not** in git. This page
records what exists on the machine that produced the reports, its size, its
SHA-256, and how to recreate it. Nothing here is required to read the source or
run the tests on fixtures.

## Trained artifacts

| Artifact | Path | Size | SHA-256 |
| --- | --- | --- | --- |
| Accelerator candidate, best checkpoint (step 14,500) | `models/accel_pilot/best.pt` | 113,348,627 B | `cf813c77caed6e7e6785318bfa8630feb70d5f26ec2887f45735a2859c126231` |
| Accelerator candidate, latest checkpoint (step 15,897) | `models/accel_pilot/checkpoint.pt` | 113,357,499 B | `89b7780af3aec778efb5fb690f4ed3e27548d3fea243aa1d4fa3cce504e6f85a` |
| Scaled experiment, best checkpoint (step 18,000) | `models/scaled_pilot/best.pt` | 238,273,233 B | `2c2f518b1f3b015fdce0cff6ef762a4a0f081643add9ec6865856ba821f22c77` |
| Scaled experiment, latest checkpoint (step 18,799) | `models/scaled_pilot/checkpoint.pt` | 238,280,025 B | `8e75f4dbc5fc77b11ff5c9b6f1f84d201534ebd15e7246b35f05f1c584f04f77` |
| Archived scaled step-14,000 best | `models/scaled_pilot/best_step14000.pt` | 238,265,169 B | `199e2f4ffb58584f08b5a0ff06b69ef02863ed61cd466b0e8f600d4bf43d7b8f` |
| Archived scaled step-14,218 latest | `models/scaled_pilot/checkpoint_step14218.pt` | 238,271,961 B | `2902bda38ae8ad53bc4dcf14210d24f646ae30d646adff7efb6898ee36ffe75a` |
| Archived step-9,500 best | `models/accel_pilot/best_step9500.pt` | 113,328,147 B | `c48eb33b6346e54ddded7a1511f7e515975ae8ce7a29749a54c5f2a9fc0d9376` |
| Archived step-9,635 latest | `models/accel_pilot/checkpoint_step9635.pt` | 113,332,923 B | `4a0bf94cc4b3aa08f950a3102b4d04a6b4ca5b866cf501a4720c7df7aaa37d70` |
| Archived step-2,000 best | `models/accel_pilot/best_step2000.pt` | 113,296,275 B | `a0a1229523f80de9badc0b740ee1c853908fa69e0fbb7379585169a058094268` |
| Archived step-2,000 latest | `models/accel_pilot/checkpoint_step2000.pt` | 113,301,051 B | `6d57e793768a4a58397d945b2050eece1791fce66d4507de051ad3f0e743f74b` |
| CPU pilot, best checkpoint (step 250) | `models/cpu_pilot/best.pt` | — | `a7cdf85df259d0596d90463cf6f812a31934e4928ab1f4a050bf022cc4e10d33` |
| Joint tokenizer (8,000 entries) | `models/cpu_small/tokenizer/` | — | `tokenizer.json` `b17a0d6732e32c0c4815c36a0e13d9904c7b512be68e5e47f4c64255c5fc1ad6` |
| Random-initialization baseline | `models/random_init/checkpoint.pt` | — | created by `scripts/make_baselines.py` |

The CPU pilot is the `cpu_small` architecture (1,950,208 parameters); the
accelerator candidate is `existing_accelerator` (9,421,824 parameters). They use
the same tokenizer, so their scores are comparable, but they are different runs.
The scaled experiment is `scaled_accelerator` (19,825,920 parameters); its
step-18,000 checkpoint is exported separately to `models/scaled_export` and is
not the default package. The step-14,000 and step-14,218 scaled checkpoints are
preserved for the archived comparison. The step-2,000 and step-9,500 accelerator
artifacts are kept for the paired comparisons in
`reports/newstest2014.comparisons.json` and
`reports/newstest2014.82m.comparisons.json`.

## Exported Java package — `models/accel_export/`

11 files, 125,419,958 B (~120 MB), produced by `nmt.cli export` from the
step-14,500 checkpoint. The previous step-9,500 package is archived at
`models/accel_export_step9500/`. `manifest.json` records every checksum below
and is verified by `Translator.open` before inference.

| File | SHA-256 |
| --- | --- |
| `onnx/encoder.onnx` | `49a5b975af1681dc1a3ffb791669e312cff0701fc21d54a3a42828df2d99ffa2` |
| `onnx/decoder_init.onnx` | `fbb33384696ad2145133d8ba5652ce909fc6af0082f47d930ebd2f59c1363709` |
| `onnx/decoder_step.onnx` | `0081c3427adac98f4e1183e6854b01356eca1d3af83ce3cfdb3d02cdb48a2a5b` |
| `model_config.json` | `d6f761e6e2c773d5ab85d676aa05016c35c59b4668d85932408a080e49959751` |
| `generation.json` | `2fb66e0d6c7d1d39bdf03f8a8368f2d70b52ac0b210d505a216798e14a29a0b1` |
| `tokenizer/tokenizer.json` | `b17a0d6732e32c0c4815c36a0e13d9904c7b512be68e5e47f4c64255c5fc1ad6` |
| `tokenizer/merges.txt` | `1b8eda9c55e92120056b2bb213a09899e677d488b1f444681ad508d6b838541e` |
| `tokenizer/merges.sha256` | `54c5c90cdf9230ebec265a61970b12f82733ae53101e3ac5e62dd9c8aaf6dca7` |
| `tokenizer/tokenizer_report.json` | `059fc82d38daceb5e8ce9766a048373ff7914a78d49dba1c8c37417b50dfe253` |
| `parity_vectors.json` | `1f14897fb3ffc5da8de4af8cdee55a695bb4d5dc0f9327a3f6d079c8202501ec` |
| `torch_state_dict.pt` | `66f9c9cbcc560d033b31987c0bffaf899bcefe09f7cfe9f174e33cd0987f3559` |

The step-9,500 package is archived at `models/accel_export_step9500/`; the
step-2,000 package remains at `models/accel_export_step2000/`. Their
`manifest.json` files record their own checksums. Rebuild the current package with:

```bash
$PY -m nmt.cli export --config configs/existing_accelerator.yaml \
  --checkpoint models/accel_pilot/best.pt --tokenizer models/cpu_small/tokenizer \
  --out models/accel_export --provenance data/prepared/europarl_nc/manifest.json
```

## Exported Java package — `models/scaled_export/`

12 files, 237,211,722 B, produced by `nmt.cli export` from the scaled step-18,000
checkpoint. It is not the default package. `manifest.json` records every
checksum below and is verified by `Translator.open` before inference.

| File | SHA-256 |
| --- | --- |
| `onnx/encoder.onnx` | `8937c77c0caac9db4eba9502ebb3acf6c4414dfdea11dc42f2cf4047ed836b7f` |
| `onnx/decoder_init.onnx` | `939a64fb1643ff815669ab6f92c1cc70c910b0d1927f44237af3f16e6f323328` |
| `onnx/decoder_step.onnx` | `76a27a26775463b970346165dbfe7b346ea85d6ecac825564cf92bc16e01bc7e` |
| `model_config.json` | `3eb10820fa8a6e408a53a8676b7205cb9fdfe592410f043893ce24cd148f86d0` |
| `generation.json` | `2fb66e0d6c7d1d39bdf03f8a8368f2d70b52ac0b210d505a216798e14a29a0b1` |
| `tokenizer/tokenizer.json` | `b17a0d6732e32c0c4815c36a0e13d9904c7b512be68e5e47f4c64255c5fc1ad6` |
| `tokenizer/merges.txt` | `1b8eda9c55e92120056b2bb213a09899e677d488b1f444681ad508d6b838541e` |
| `tokenizer/merges.sha256` | `54c5c90cdf9230ebec265a61970b12f82733ae53101e3ac5e62dd9c8aaf6dca7` |
| `tokenizer/tokenizer_report.json` | `059fc82d38daceb5e8ce9766a048373ff7914a78d49dba1c8c37417b50dfe253` |
| `parity_vectors.json` | `a632661604cd16a243f536d362ba822f184d2693b0ceff823677408ee24e3461` |
| `torch_state_dict.pt` | `775dde0f9d847763317e95f277c11165f6ca6e2229bae9c38febeb430e0cfc79` |

Python↔ONNX parity passed with worst max-absolute logit difference 6.44e-6 and
identical greedy generation. The Java suite passed 10/10 tests. Java benchmark:
cold load 1,542 ms; first translation 508 ms; warm median 354 ms, p95 562 ms;
batch of 5 inputs 1,731 ms (2.89 inputs/s). Rebuild with:

```bash
$PY -m nmt.cli export --config configs/scaled_accelerator.yaml \
  --checkpoint models/scaled_pilot/best.pt --tokenizer models/cpu_small/tokenizer \
  --out models/scaled_export --provenance data/prepared/europarl_nc/manifest.json
$PY -m nmt.cli parity --model-dir models/scaled_export \
  --out reports/parity_scaled.json
```

## Corpora

| Archive | Path | SHA-256 |
| --- | --- | --- |
| Europarl v7 FR–EN | `data/raw/archives/` | `80f5d52113e23bfe51f890569ed69d2a1348127bca0894d07d1d23e0d0fac1e9` |
| WMT14 News Commentary v9 | `data/raw/archives/` | `cb8953f292298e6877ae433c98912b927cb0766b303f4540512ddd286c748485` |
| WMT13 test sources (unused for training) | `data/raw/archives/` | `e87fb37e2a6b9f11d39f7771ee7d354343184a5f3a8e518942451e49cbf4c186` |
| WMT14 `full` test set | `data/raw/archives/` | `f298b2435218a563268da99a66b21c03d925895c5f26fac82706e12b40b63619` |

Prepared splits and their manifests live in `data/prepared/europarl_nc/` and
`data/prepared/newstest2014/`. They can be regenerated with
`nmt.cli prepare --source all`; acquisition verifies the archive checksums and
refuses to exceed the configured download/storage caps.

## Reports

Reports include `hardware.json`, `parity.json`, `java_benchmark.json`,
`parity_step9500.json`, `java_benchmark_step9500.json`,
`parity_step15897.json`, `java_benchmark_step15897.json`,
`newstest2014.82m.{en-fr,fr-en}.full.json` and their `_summary.json`
companions, `newstest2014.82m.comparisons.json`, the earlier
`newstest2014.{en-fr,fr-en}.full.json` and `newstest2014.comparisons.json`, the
archived `newstest2014.{en-fr,fr-en}.step2000*.json` reports,
`subset600.en-fr.json`/`subset600.en-fr_summary.json` (random/CPU/accelerator
baselines on a documented 564-row EN→FR subset), the archived review packets
and AI-assisted results in `reports/review_packet_enfr/`,
`reports/review_packet_fren/` (candidate step 9,500), and
`reports/review_summary.{json,md}`, plus the fresh step-14,500 packets/results
in `reports/review_packet_82m_enfr/`, `reports/review_packet_82m_fren/`, and
`reports/review_summary_82m.{json,md}` (43/200 acceptable; **not** human
evaluation), the beam-vs-greedy sample reports
`reports/beam_vs_greedy_step14500.{json,md}` with the per-direction greedy and
beam-4 raw JSON, the dev tuning report `reports/beam_tuning_dev.md` with its raw
sweep directory, and the full tuned comparison
`reports/beam_tuned_step14500.{json,md}` with the per-direction full JSON. The
machine-assisted API spot check currently uses the final step-15,897 checkpoint
in `reports/review_api_google_82m_final_enfr/` and
`reports/review_api_google_82m_final_fren/` (per-row TSV, summary JSON,
Markdown, and local API caches; **not** human review); the earlier step-14,500
comparison remains under `reports/review_api_google_82m_enfr/` and
`reports/review_api_google_82m_fren/`. The final-only 25-row inputs are under
`reports/gtx_final_step15897_inputs/`. The final-checkpoint reports are
`reports/newstest2014.82m.final.{en-fr,fr-en}.full.json` with summaries, plus
`reports/newstest2014.82m.final.comparisons.json` and `.md` for the combined
source-copy and checkpoint comparisons. The full final-beam reports are
`reports/newstest2014.82m.final_beam4.{en-fr,fr-en}.full.json` with summaries,
plus `reports/newstest2014.82m.final_beam4.comparisons.json` and `.md`.

The scaling experiment is configured by `configs/scaled_accelerator.yaml` and
launched by `scripts/run_scaled_pilot.sh`; it writes under
`models/scaled_pilot/`. Its step-18,000 best checkpoint and step-18,799 latest
checkpoint are the two current scaled rows above; the step-14,000 and step-14,218
rows are archived. The current scaled full-set reports are
`reports/newstest2014.scaled_resumed.{greedy,beam4}.{en-fr,fr-en}.full.json`
with summaries, plus `reports/newstest2014.scaled_resumed.comparisons.{json,md}`;
the archived step-14,000 reports remain at `reports/newstest2014.scaled.*`. The
scaled export is `models/scaled_export/`; parity and Java evidence is in
`reports/parity_scaled.json`, `reports/parity_scaled.log`,
`reports/java_build_scaled.log`, and `reports/java_benchmark_scaled.json`. The
per-direction evaluation JSON is written by the commands named in `README.md`;
`newstest2014.82m.comparisons.json` is a combined view assembled from the two
`_summary.json` files. Training history and checkpoints are under
`models/accel_pilot/` and `models/scaled_pilot/`; exported packages are under
`models/accel_export/` (default step 14,500),
`models/accel_export_step9500/`, `models/accel_export_step15897/` (final small
verification), and `models/scaled_export/` (scaled step 18,000).

The small final training report records `stop_reason = max_train_tokens`, with
the continuation log in `reports/train_continue_82m.log`; the scaled report
also records `stop_reason = max_train_tokens`, at step 18,799 / 164,002,778
tokens. Its log is `reports/scaled_resume.log`.

## Terms

These artifacts are derived from Europarl v7 and WMT14 News Commentary v9; the
corpora's terms are not permissive by default and are recorded in `NOTICE`,
`docs/DATA_CARD.md` and `docs/MODEL_CARD.md`. Model weights are released for
research use under the same constraints; no blanket permissive license is
claimed for them.
