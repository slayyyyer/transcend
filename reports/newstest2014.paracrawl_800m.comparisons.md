# ParaCrawl 800M arm vs full-data 800M checkpoint on newstest2014

Baseline: `models\scaled_full\best.pt` (label `full800`, step 77,000; SHA-256 `44f963e715121d4efd4746a8631a674808d899b3ff72cd6cb9871649ec68de86`).  
Candidate `paracrawl_800m`: `models\data_ablation\paracrawl_800m\best.pt` (step 91,000; SHA-256 `4e578110c728465a708b357708a5dbbb640180308aa86a4922a584bec088575e`).  
Training report: step 91,754, 800,004,736 tokens, stop reason `max_train_tokens`.  
All checkpoints were evaluated on exactly the same retained newstest2014 rows with greedy and beam-4 (length penalty 1.2, minimum 16 generated target tokens). Positive deltas mean the candidate checkpoint is better.

| Direction | Decoder | Candidate | Candidate step | Baseline BLEU / chrF++ | Candidate BLEU / chrF++ | ΔBLEU | ΔchrF++ | chrF++ 95% CI | Sentence W/L/T |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| en-fr | greedy | paracrawl_800m | 91,000 | 25.021 / 51.065 | 26.174 / 52.144 | +1.153 | +1.079 | [+0.757, +1.388] | 1604/1196/159 |
| en-fr | beam4 | paracrawl_800m | 91,000 | 26.277 / 52.249 | 27.709 / 53.260 | +1.432 | +1.011 | [+0.743, +1.283] | 1606/1167/186 |
| fr-en | greedy | paracrawl_800m | 91,000 | 23.330 / 50.600 | 24.806 / 51.685 | +1.476 | +1.085 | [+0.843, +1.334] | 1569/1162/191 |
| fr-en | beam4 | paracrawl_800m | 91,000 | 23.898 / 51.462 | 25.458 / 52.565 | +1.560 | +1.103 | [+0.903, +1.324] | 1583/1144/195 |

Retained rows: 2,959/3,003 en-fr (44 over the 128-token source limit); 2,922/3,003 fr-en (81 over the 128-token source limit).

Paired bootstrap resamples the 176 newstest2014 documents (1,000 resamples). Sentence W/L/T uses sentence-level chrF++; it is descriptive, not a significance test. These are automatic metrics only, not human evaluation.
