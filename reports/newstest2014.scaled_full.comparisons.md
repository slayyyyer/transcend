# Full-data checkpoint vs scaled step 18,000 on newstest2014

Candidate: `models/scaled_full/best.pt` (step 38,000; SHA-256 `fee0f72cfc4556f03fd6820157cc9c530c41f0a6db67153f8185d1e92b813277`).  
Baseline: `models/scaled_pilot/best.pt` (step 18,000; SHA-256 `2c2f518b1f3b015fdce0cff6ef762a4a0f081643add9ec6865856ba821f22c77`).  
Both checkpoints were evaluated on exactly the same retained newstest2014 rows with greedy and beam-4 (length penalty 1.2, minimum 16 generated target tokens). Positive deltas mean the full-data checkpoint is better.

| Direction | Decoder | n | Step 18k BLEU / chrF++ | Full-data BLEU / chrF++ | ΔBLEU | ΔchrF++ | BLEU 95% CI | chrF++ 95% CI | Sentence chrF++ W/L/T |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| en-fr | greedy | 2959 | 20.011 / 47.036 | 22.655 / 49.045 | +2.644 | +2.009 | [+2.185, +3.090] | [+1.742, +2.302] | 1819/1065/75 |
| en-fr | beam4 | 2959 | 21.243 / 48.442 | 24.079 / 50.318 | +2.836 | +1.876 | [+2.282, +3.424] | [+1.633, +2.113] | 1780/1084/95 |
| fr-en | greedy | 2922 | 19.018 / 46.698 | 21.260 / 48.755 | +2.242 | +2.057 | [+1.874, +2.617] | [+1.783, +2.310] | 1823/1022/77 |
| fr-en | beam4 | 2922 | 20.300 / 48.152 | 22.145 / 49.882 | +1.845 | +1.730 | [+1.468, +2.228] | [+1.496, +1.977] | 1773/1047/102 |

Retained rows: 2,959/3,003 en-fr (44 over the 128-token source limit); 2,922/3,003 fr-en (81 over the 128-token source limit).

Paired bootstrap resamples the 176 newstest2014 documents (1,000 resamples). Sentence W/L/T uses sentence-level chrF++; it is descriptive, not a significance test. These are automatic metrics only, not human evaluation.
