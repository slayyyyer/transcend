# Final step-15,897 beam 4 on full newstest2014

Checkpoint: step 15,897 (`models/accel_pilot/checkpoint.pt`). Beam 4, length penalty 1.2 and minimum 16 new tokens, using the settings selected on dev for step 14,500. The baseline is the final greedy run on exactly the same retained segments and references.

| Direction | Decoder | n | BLEU | chrF++ | mean words | ≤5 words | ≤8 words |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| en-fr | greedy | 2959 | 12.110 | 39.459 | 24.731 | 38 | 191 |
| en-fr | beam | 2959 | 14.455 | 41.365 | 23.906 | 10 | 189 |
| fr-en | greedy | 2922 | 12.169 | 39.528 | 22.854 | 58 | 212 |
| fr-en | beam | 2922 | 14.113 | 41.452 | 22.328 | 7 | 206 |

| Direction | ΔBLEU | ΔchrF++ | paired BLEU Δ [95% CI] | paired chrF++ Δ [95% CI] | sentence chrF++ W/L/T | changed / shorter / longer |
| --- | ---: | ---: | --- | --- | --- | --- |
| en-fr | +2.345 | +1.906 | +2.345 [+2.076, +2.612] | +1.907 [+1.684, +2.125] | 1786/1045/128 | 2834/1386/970 |
| fr-en | +1.944 | +1.924 | +1.944 [+1.691, +2.223] | +1.924 [+1.734, +2.117] | 1729/1033/160 | 2764/1300/975 |

Paired bootstrap resamples the 176 newstest2014 documents (1,000 resamples). Sentence W/L/T counts use sentence-level chrF++; they are descriptive, not a significance test. This is an automatic metric comparison, not a human evaluation. Greedy remains the default decoder.
