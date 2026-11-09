# Full-data 800M continuation vs 350M checkpoint on newstest2014

Baseline: `models\scaled_full\best.pre_extend.pt` (label `full350`, step 38,000; SHA-256 `fee0f72cfc4556f03fd6820157cc9c530c41f0a6db67153f8185d1e92b813277`).  
Candidate `full800best`: `models\scaled_full\best.pt` (step 77,000; SHA-256 `44f963e715121d4efd4746a8631a674808d899b3ff72cd6cb9871649ec68de86`).  
Candidate `full800final`: `models\scaled_full\checkpoint.pt` (step 91,691; SHA-256 `6fe833dd7c359d62c01935b5d7f117e2c490a69b83decf3550da7af059741342`).  
Training report: step 91,691, 800,002,117 tokens, stop reason `max_train_tokens`.  
All checkpoints were evaluated on exactly the same retained newstest2014 rows with greedy and beam-4 (length penalty 1.2, minimum 16 generated target tokens). Positive deltas mean the continuation checkpoint is better.

| Direction | Decoder | Candidate | Candidate step | Baseline BLEU / chrF++ | Candidate BLEU / chrF++ | ΔBLEU | ΔchrF++ | chrF++ 95% CI | Sentence W/L/T |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| en-fr | greedy | full800best | 77,000 | 22.655 / 49.045 | 25.021 / 51.065 | +2.366 | +2.020 | [+1.788, +2.269] | 1766/1029/164 |
| en-fr | greedy | full800final | 91,691 | 22.655 / 49.045 | 25.367 / 51.391 | +2.712 | +2.346 | [+2.106, +2.587] | 1851/978/130 |
| en-fr | beam4 | full800best | 77,000 | 24.079 / 50.318 | 26.277 / 52.249 | +2.198 | +1.931 | [+1.745, +2.113] | 1743/992/224 |
| en-fr | beam4 | full800final | 91,691 | 24.079 / 50.318 | 26.440 / 52.531 | +2.361 | +2.213 | [+1.998, +2.451] | 1814/937/208 |
| fr-en | greedy | full800best | 77,000 | 21.260 / 48.755 | 23.330 / 50.600 | +2.070 | +1.845 | [+1.657, +2.043] | 1719/994/209 |
| fr-en | greedy | full800final | 91,691 | 21.260 / 48.755 | 23.558 / 50.830 | +2.298 | +2.075 | [+1.859, +2.282] | 1766/944/212 |
| fr-en | beam4 | full800best | 77,000 | 22.145 / 49.882 | 23.898 / 51.462 | +1.753 | +1.580 | [+1.384, +1.781] | 1634/1016/272 |
| fr-en | beam4 | full800final | 91,691 | 22.145 / 49.882 | 24.377 / 51.902 | +2.232 | +2.020 | [+1.801, +2.253] | 1740/948/234 |

Retained rows: 2,959/3,003 en-fr (44 over the 128-token source limit); 2,922/3,003 fr-en (81 over the 128-token source limit).

Paired bootstrap resamples the 176 newstest2014 documents (1,000 resamples). Sentence W/L/T uses sentence-level chrF++; it is descriptive, not a significance test. These are automatic metrics only, not human evaluation.
