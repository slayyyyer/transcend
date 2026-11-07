# API-assisted review — fr-en

This is **machine-assisted review**, not a human evaluation. The API translations are a third machine opinion; they are not gold references.

Sample: 25 segments selected deterministically (seed 20261107) with the same diagnostic-marker stratification as the blind review packet.
Checkpoint: `final` step 15897 (greedy); `final` step 15897 (tuned beam).
API: Google Translate (gtx, unofficial), 25 network calls, 0 cache hits, 25 rows usable. Long sources were split at 450 UTF-8 bytes.

| System | BLEU | chrF++ | mean words |
| --- | ---: | ---: | ---: |
| Greedy | 18.341 | 43.487 | 15.800 |
| Tuned beam | 21.780 | 45.658 | 15.240 |
| Google Translate (gtx, unofficial) | 49.329 | 68.008 | 14.680 |

| Comparison | candidate wins | baseline wins | ties |
| --- | ---: | ---: | ---: |
| Beam vs greedy | 16 | 6 | 3 |
| Google Translate (gtx, unofficial) vs greedy | 25 | 0 | 0 |
| Google Translate (gtx, unofficial) vs beam | 25 | 0 | 0 |

API status counts: `ok`=25.

The Google gtx endpoint is free and needs no key, but it is an undocumented public endpoint with no SLA and may rate-limit or change. The harness caches responses and sleeps between requests. The cache path is `reports\review_api_google_82m_final_fren\api_cache.json`.

Caveats: the sample is not the full benchmark; chunking can degrade translations; API output is not a human judgment; and no human evaluation is claimed by this report.
