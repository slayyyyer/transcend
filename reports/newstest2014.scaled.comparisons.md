# Scaled checkpoint on full newstest2014

Checkpoint: `models/scaled_pilot/best.pt`, selected by the 300-row-per-direction dev validation. Baseline: the final step-15,897 small model. Greedy and beam-4 (length penalty 1.2, minimum 16 generated target tokens). Both sides use the same retained newstest2014 segments and references.

| Direction | Comparison | n | Baseline BLEU / chrF++ | Scaled BLEU / chrF++ | ΔBLEU | ΔchrF++ | chrF++ 95% CI | Sentence W/L/T | changed / shorter / longer |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| en-fr | scaled greedy vs final greedy | 2959 | 12.110 / 39.459 | 18.924 / 45.729 | +6.814 | +6.270 | [+5.961, +6.581] | 2327/611/21 | 2939/1619/934 |
| en-fr | scaled beam-4 vs final beam-4 | 2959 | 14.455 / 41.365 | 20.670 / 47.216 | +6.215 | +5.851 | [+5.564, +6.143] | 2301/639/19 | 2940/1463/1004 |
| en-fr | scaled beam-4 vs scaled greedy | 2959 | 18.924 / 45.729 | 20.670 / 47.216 | +1.746 | +1.487 | [+1.294, +1.671] | 1658/1026/275 | 2698/1122/952 |
| fr-en | scaled greedy vs final greedy | 2922 | 12.169 / 39.528 | 17.858 / 45.415 | +5.689 | +5.887 | [+5.589, +6.185] | 2256/645/21 | 2903/1680/806 |
| fr-en | scaled beam-4 vs final beam-4 | 2922 | 14.113 / 41.452 | 19.414 / 46.915 | +5.301 | +5.463 | [+5.231, +5.694] | 2254/641/27 | 2896/1546/857 |
| fr-en | scaled beam-4 vs scaled greedy | 2922 | 17.858 / 45.415 | 19.414 / 46.915 | +1.556 | +1.500 | [+1.315, +1.689] | 1577/1003/342 | 2588/958/1022 |

Paired bootstrap resamples the 176 newstest2014 documents (1,000 resamples). Sentence W/L/T uses sentence-level chrF++; it is descriptive, not a significance test. All numbers are automatic metrics, not human evaluation.
