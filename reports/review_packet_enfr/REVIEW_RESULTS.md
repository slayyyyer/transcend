# AI-assisted review results — en-fr

**Status: AI-assisted review complete; qualified human review pending.** This is
not a human evaluation and must not be cited as one.

* Packet: `review_en-fr.tsv` (100 rows; deterministic sample from 2,959 segments,
  seed 20261107).
* Reviewed file: `review_en-fr_reviewed.tsv` (fills the four reviewer columns).
* Reviewer: Buffy, AI coding agent, using the source, reference, system output
  and diagnostic flags. The packet did not display the system identity, but this
  was not a blinded human review.
* Rubric: OK / MINOR / MAJOR / CRITICAL for meaning preservation; acceptable is
  OK + MINOR.
* Results: **1 OK, 14 MINOR, 68 MAJOR, 17 CRITICAL**; **15/100 acceptable (15%)**.
* Rows by error category (overlapping): mistranslation 85, grammar 59,
  terminology 52, omission 36, repetition 24, agreement 4, punctuation 3.
* Diagnostic-marker slice: 25 rows, 1 acceptable; unflagged: 75 rows,
  14 acceptable.

The original `review_en-fr.tsv` remains unmodified with empty reviewer columns.
Aggregate results and methodology are in `../review_summary.md` and
`../review_summary.json`. A qualified bilingual reviewer has **not** scored this
packet; human evaluation remains pending.
