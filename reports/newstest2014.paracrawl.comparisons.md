# ParaCrawl 350M arm vs full-data 350M checkpoint on newstest2014

Baseline: `models\scaled_full\best.pre_extend.pt` (label `full350`, step 38,000; SHA-256 `fee0f72cfc4556f03fd6820157cc9c530c41f0a6db67153f8185d1e92b813277`).  
Candidate `paracrawl`: `models\data_ablation\paracrawl\best.pt` (step 40,000; SHA-256 `bf4cd44e429e0b5d84f3eba71afa76b4b8b6a923c9fff63872d13e952ef45e32`).  
Training report: step 40,144, 350,001,418 tokens, stop reason `max_train_tokens`.  
All checkpoints were evaluated on exactly the same retained newstest2014 rows with greedy and beam-4 (length penalty 1.2, minimum 16 generated target tokens). Positive deltas mean the candidate checkpoint is better.

| Direction | Decoder | Candidate | Candidate step | Baseline BLEU / chrF++ | Candidate BLEU / chrF++ | ΔBLEU | ΔchrF++ | chrF++ 95% CI | Sentence W/L/T |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| en-fr | greedy | paracrawl | 40,000 | 22.655 / 49.045 | 23.968 / 50.062 | +1.313 | +1.017 | [+0.770, +1.275] | 1578/1267/114 |
| en-fr | beam4 | paracrawl | 40,000 | 24.079 / 50.318 | 25.220 / 51.200 | +1.141 | +0.882 | [+0.651, +1.119] | 1577/1211/171 |
| fr-en | greedy | paracrawl | 40,000 | 21.260 / 48.755 | 22.488 / 49.677 | +1.228 | +0.922 | [+0.661, +1.174] | 1589/1193/140 |
| fr-en | beam4 | paracrawl | 40,000 | 22.145 / 49.882 | 23.281 / 50.770 | +1.136 | +0.888 | [+0.618, +1.146] | 1552/1203/167 |

Retained rows: 2,959/3,003 en-fr (44 over the 128-token source limit); 2,922/3,003 fr-en (81 over the 128-token source limit).

Paired bootstrap resamples the 176 newstest2014 documents (1,000 resamples). Sentence W/L/T uses sentence-level chrF++; it is descriptive, not a significance test. These are automatic metrics only, not human evaluation.
