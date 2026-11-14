# Tatoeba 350M arm vs full-data 350M checkpoint on newstest2014

Baseline: `models\scaled_full\best.pre_extend.pt` (label `full350`, step 38,000; SHA-256 `fee0f72cfc4556f03fd6820157cc9c530c41f0a6db67153f8185d1e92b813277`).  
Candidate `tatoeba`: `models\data_ablation\tatoeba\best.pt` (step 39,000; SHA-256 `6eabf9a2831f98cd852e72ccac7106a9504286ea622e66477e2e4b7323186813`).  
Training report: step 40,149, 350,000,282 tokens, stop reason `max_train_tokens`.  
All checkpoints were evaluated on exactly the same retained newstest2014 rows with greedy and beam-4 (length penalty 1.2, minimum 16 generated target tokens). Positive deltas mean the Tatoeba candidate is better.

| Direction | Decoder | Candidate | Candidate step | Baseline BLEU / chrF++ | Candidate BLEU / chrF++ | ΔBLEU | ΔchrF++ | chrF++ 95% CI | Sentence W/L/T |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| en-fr | greedy | tatoeba | 39,000 | 22.655 / 49.045 | 22.828 / 49.224 | +0.173 | +0.179 | [-0.076, +0.435] | 1475/1386/98 |
| en-fr | beam4 | tatoeba | 39,000 | 24.079 / 50.318 | 24.224 / 50.319 | +0.145 | +0.001 | [-0.239, +0.230] | 1419/1399/141 |
| fr-en | greedy | tatoeba | 39,000 | 21.260 / 48.755 | 21.803 / 49.178 | +0.543 | +0.423 | [+0.184, +0.665] | 1492/1294/136 |
| fr-en | beam4 | tatoeba | 39,000 | 22.145 / 49.882 | 22.481 / 50.233 | +0.336 | +0.351 | [+0.118, +0.568] | 1449/1320/153 |

**Verdict:** Tatoeba improves FR→EN in both greedy and beam-4 with 95% chrF++ intervals excluding zero. EN→FR is slightly positive but not statistically significant in either decoder. No cell regresses. These are automatic newstest2014 metrics, not human evaluation.

Retained rows: 2,959/3,003 en-fr (44 over the 128-token source limit); 2,922/3,003 fr-en (81 over the 128-token source limit).

Paired bootstrap resamples the 176 newstest2014 documents (1,000 resamples). Sentence W/L/T uses sentence-level chrF++; it is descriptive, not a significance test. These are automatic metrics only, not human evaluation.
