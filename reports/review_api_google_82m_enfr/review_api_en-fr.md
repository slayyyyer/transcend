# API-assisted review — en-fr

This is **machine-assisted review**, not a human evaluation. The API translations are a third machine opinion; they are not gold references.

Sample: 25 segments selected deterministically (seed 20261107) with the same diagnostic-marker stratification as the blind review packet.
API: Google Translate (gtx, unofficial), 20 network calls, 5 cache hits, 25 rows usable. Long sources were split at 450 UTF-8 bytes.

| System | BLEU | chrF++ | mean words |
| --- | ---: | ---: | ---: |
| Greedy | 3.996 | 29.785 | 17.720 |
| Tuned beam | 4.662 | 30.414 | 16.560 |
| Google Translate (gtx, unofficial) | 31.149 | 56.128 | 13.880 |

| Comparison | candidate wins | baseline wins | ties |
| --- | ---: | ---: | ---: |
| Beam vs greedy | 17 | 7 | 1 |
| Google Translate (gtx, unofficial) vs greedy | 25 | 0 | 0 |
| Google Translate (gtx, unofficial) vs beam | 25 | 0 | 0 |

API status counts: `ok`=20, `cache`=5.

The Google gtx endpoint is free and needs no key, but it is an undocumented public endpoint with no SLA and may rate-limit or change. The harness caches responses and sleeps between requests. The cache path is `reports\review_api_google_82m_enfr\api_cache.json`.

Caveats: the sample is not the full benchmark; chunking can degrade translations; API output is not a human judgment; and no human evaluation is claimed by this report.
