# Final newstest2014 comparison

This report compares the dev-selected step-14,500 candidate with the final step-15,897
checkpoint on newstest2014. The final checkpoint was evaluated after selection and was
not tuned on this benchmark. Scores are greedy decoding, detokenized output, SacreBLEU
2.5.1 (`13a`, cased) and chrF++ (`word_order=2`).

## Scores

| System | BLEU | chrF++ |
| --- | ---: | ---: |
| en-fr source copy | 2.358 | 21.875 |
| en-fr step 9,500 | 9.214 | 35.564 |
| en-fr step 14,500 candidate | 11.031 | 38.667 |
| en-fr step 15,897 final | 12.110 | 39.459 |
| fr-en source copy | 2.378 | 23.722 |
| fr-en step 9,500 | 9.598 | 35.940 |
| fr-en step 14,500 candidate | 11.278 | 38.855 |
| fr-en step 15,897 final | 12.169 | 39.528 |

## Paired document bootstrap

| Comparison | ΔchrF++ [95% CI] |
| --- | ---: |
| en-fr final vs candidate | +0.792 [0.582, 1.012] |
| en-fr candidate vs source copy | +16.793 [15.987, 17.642] |
| en-fr final vs source copy | +17.584 [16.867, 18.341] |
| fr-en final vs candidate | +0.673 [0.467, 0.884] |
| fr-en candidate vs source copy | +15.133 [14.347, 15.932] |
| fr-en final vs source copy | +15.806 [15.040, 16.608] |

The intervals use 1,000 paired resamples over 176 documents. Positive deltas favor the
first system. The final checkpoint improves on both the candidate and the source-copy
baseline in both directions; this is an automatic-metric result, not human review.
