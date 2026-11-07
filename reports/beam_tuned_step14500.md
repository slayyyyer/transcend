# Tuned beam search on full newstest2014

Checkpoint: step 14,500 (`models/accel_pilot/best.pt`). Beam 4, length penalty 1.2, minimum 16 generated target tokens, selected on dev only (see `reports/beam_tuning_dev.md`). Greedy is the archived full-set baseline; both runs use exactly the same retained segments and references.

| Direction | Decoder | n | BLEU | chrF++ | mean words | ≤5 words | ≤8 words |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| en-fr | greedy | 2959 | 11.031 | 38.667 | 26.088 | 32 | 158 |
| en-fr | beam | 2959 | 13.417 | 40.817 | 24.918 | 8 | 168 |
| fr-en | greedy | 2922 | 11.278 | 38.855 | 23.863 | 40 | 174 |
| fr-en | beam | 2922 | 13.147 | 40.881 | 23.184 | 4 | 182 |

| Direction | ΔBLEU | ΔchrF++ | paired BLEU Δ [95% CI] | paired chrF++ Δ [95% CI] | sentence chrF++ W/L/T | changed / shorter / longer |
| --- | ---: | ---: | --- | --- | --- | --- |
| en-fr | +2.386 | +2.150 | +2.387 [+2.153, +2.633] | +2.150 [+1.961, +2.337] | 1835/1017/107 | 2856/1519/926 |
| fr-en | +1.869 | +2.026 | +1.869 [+1.593, +2.130] | +2.026 [+1.805, +2.263] | 1797/984/141 | 2784/1365/964 |

Paired bootstrap resamples the 176 newstest2014 documents (1,000 resamples) and is one-sided for the p-value shown in the JSON. Sentence W/L/T counts use sentence-level chrF++; they are descriptive and not a significance test.

The minimum-length setting prevents EOS before 16 generated target tokens. It reduces very short outputs but does not remove all mistranslations or truncation-like omissions; 1,519 EN→FR and 1,365 FR→EN outputs are shorter than greedy even after the floor. This is an automatic metric comparison, not a human evaluation; qualified human review remains pending. Greedy remains the default decoder.
