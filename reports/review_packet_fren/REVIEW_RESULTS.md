# AI-assisted review results — fr-en

**Status: AI-assisted review complete; qualified human review pending.** This is
not a human evaluation and must not be cited as one.

* Packet: `review_fr-en.tsv` (100 rows; deterministic sample from 2,922 segments,
  seed 20261107).
* Reviewed file: `review_fr-en_reviewed.tsv` (fills the four reviewer columns).
* Reviewer: Buffy, AI coding agent, using the source, reference, system output
  and diagnostic flags. The packet did not display the system identity, but this
  was not a blinded human review.
* Rubric: OK / MINOR / MAJOR / CRITICAL for meaning preservation; acceptable is
  OK + MINOR.
* Results: **1 OK, 17 MINOR, 74 MAJOR, 8 CRITICAL**; **18/100 acceptable (18%)**.
* Rows by error category (overlapping): mistranslation 87, terminology 65,
  omission 36, grammar 26, repetition 13, agreement 3, addition 1.
* Diagnostic-marker slice: 25 rows, 2 acceptable; unflagged: 75 rows,
  16 acceptable.

The original `review_fr-en.tsv` remains unmodified with empty reviewer columns.
Aggregate results and methodology are in `../review_summary.md` and
`../review_summary.json`. A qualified bilingual reviewer has **not** scored this
packet; human evaluation remains pending.
