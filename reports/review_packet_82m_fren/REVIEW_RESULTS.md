# AI-assisted review results — FR→EN

**Status: AI-assisted review complete; qualified human review pending.** This is
not a human evaluation and must not be cited as one.

* Packet: `review_packet_82m_fren/review_fr-en.tsv` (100 rows; deterministic
  sample, seed 20261107).
* Reviewed file: `review_packet_82m_fren/review_fr-en_reviewed.tsv`.
* Reviewer: Buffy, AI coding agent, using source, reference, system output and
  diagnostic flags. The packet did not display the model identity, but this was
  not a blinded human review.
* Results: **1 OK, 23 MINOR, 54 MAJOR, 22 CRITICAL**; **24/100 acceptable
  (24.0%)**.
* Error categories (overlapping): mistranslation 91, terminology 71, omission 61,
  grammar 54, repetition 9, agreement 1, addition 1.
* Diagnostic-marker slice: 6/25 flagged rows acceptable; 18/75 unflagged rows
  acceptable.
* Archived step-9,500 result for the same sample: 18/100 acceptable. The fresh
  review has more CRITICAL labels, so the net count is not a uniform improvement.

The original `review_fr-en.tsv` remains unchanged with empty reviewer columns.
Aggregate results and methodology are in `../review_summary_82m.md` and
`../review_summary_82m.json`. A qualified bilingual reviewer has **not** scored
this packet; human evaluation remains pending.
