# Side-by-side review packet — full-data model

**20 segments: 10 EN→FR and 10 FR→EN.**

- Model: `models/scaled_full/best.pt`, step 38,000 (SHA-256 `fee0f72cfc4556f03fd6820157cc9c530c41f0a6db67153f8185d1e92b813277`).
- Decoding: beam 4, length penalty 1.2, minimum 16 new tokens.
- Sources: `reports\newstest2014.scaled_full.beam4.en-fr.full.json` and `reports\newstest2014.scaled_full.beam4.fr-en.full.json`.
- Sampling: deterministic SHA-256 ranking with seed `20261107`; 2 diagnostic-marker
  segments per direction and 8 general segments per direction.
- Reviewer judgments: **pending**. No OK/MINOR/MAJOR/CRITICAL labels have been
  filled in.

## Files

- `review_packet.tsv` — review form with blank reviewer columns.
- `review_packet.md` — read-only side-by-side view.
- `manifest.json` — provenance, sample IDs, report hashes and row mapping.

## Instructions

1. Mark each row OK / MINOR / MAJOR / CRITICAL for meaning preservation.
2. Record error categories in the free-text column: omission, addition,
   mistranslation, grammar, agreement, terminology, punctuation.
3. If another wording is acceptable, write it in the last column.
4. Save the completed copy as `review_packet_reviewed.tsv`; keep the original
   blank form unchanged.
5. A round-trip translation is not evidence of correctness.

For a blinded review, give the reviewer only `review_packet.tsv`; the README and
manifest identify the system for provenance.

Human evaluation status: **pending** until a qualified bilingual reviewer
completes the packet. No reviewer judgments are fabricated here.
