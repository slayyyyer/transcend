# AI-assisted review results — EN→FR

**Status: AI-assisted review complete; qualified human review pending.** This is
not a human evaluation and must not be cited as one.

* Packet: `review_packet_82m_enfr/review_en-fr.tsv` (100 rows; deterministic
  sample, seed 20261107).
* Reviewed file: `review_packet_82m_enfr/review_en-fr_reviewed.tsv`.
* Reviewer: Buffy, AI coding agent, using source, reference, system output and
  diagnostic flags. The packet did not display the model identity, but this was
  not a blinded human review.
* Results: **2 OK, 17 MINOR, 53 MAJOR, 28 CRITICAL**; **19/100 acceptable
  (19.0%)**.
* Error categories (overlapping): mistranslation 88, terminology 48, grammar 73,
  omission 61, repetition 10, agreement 3.
* Diagnostic-marker slice: 2/25 flagged rows acceptable; 17/75 unflagged rows
  acceptable.
* Archived step-9,500 result for the same sample: 15/100 acceptable. The fresh
  review has more CRITICAL labels, so the net count is not a uniform improvement.

The original `review_en-fr.tsv` remains unchanged with empty reviewer columns.
Aggregate results and methodology are in `../review_summary_82m.md` and
`../review_summary_82m.json`. A qualified bilingual reviewer has **not** scored
this packet; human evaluation remains pending.
