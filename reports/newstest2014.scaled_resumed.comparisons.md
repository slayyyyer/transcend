# Resumed scaled checkpoint on full newstest2014

Checkpoint: `models/scaled_pilot/best.pt`, selected by the 300-row-per-direction dev validation. Baseline: the final step-15,897 small model. Greedy and beam-4 (length penalty 1.2, minimum 16 generated target tokens). Both sides use the same retained newstest2014 segments and references.

| Direction | Comparison | n | Baseline BLEU / chrF++ | Scaled BLEU / chrF++ | ΔBLEU | ΔchrF++ | chrF++ 95% CI | Sentence W/L/T | changed / shorter / longer |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| en-fr | scaled greedy vs final greedy | 2959 | 12.110 / 39.459 | 20.011 / 47.036 | +7.901 | +7.577 | [+7.264, +7.896] | 2426/510/23 | 2937/1536/996 |
| en-fr | scaled beam-4 vs final beam-4 | 2959 | 14.455 / 41.365 | 21.243 / 48.442 | +6.788 | +7.077 | [+6.785, +7.373] | 2431/507/21 | 2938/1320/1192 |
| en-fr | scaled beam-4 vs scaled greedy | 2959 | 20.011 / 47.036 | 21.243 / 48.442 | +1.232 | +1.406 | [+1.206, +1.610] | 1621/1049/289 | 2681/1032/960 |
| fr-en | scaled greedy vs final greedy | 2922 | 12.169 / 39.528 | 19.018 / 46.698 | +6.849 | +7.170 | [+6.863, +7.490] | 2385/520/17 | 2906/1593/883 |
| fr-en | scaled beam-4 vs final beam-4 | 2922 | 14.113 / 41.452 | 20.300 / 48.152 | +6.187 | +6.700 | [+6.424, +6.982] | 2375/524/23 | 2903/1406/967 |
| fr-en | scaled beam-4 vs scaled greedy | 2922 | 19.018 / 46.698 | 20.300 / 48.152 | +1.282 | +1.454 | [+1.280, +1.625] | 1529/995/398 | 2533/896/1021 |

Paired bootstrap resamples the 176 newstest2014 documents (1,000 resamples). Sentence W/L/T uses sentence-level chrF++; it is descriptive, not a significance test. All numbers are automatic metrics, not human evaluation.
