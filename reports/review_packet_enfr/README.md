# Blind bilingual review packet — en-fr

* Source: `data/prepared/newstest2014/newstest2014.en-fr.jsonl` (2959 segments, 176 documents).
* System: `candidate` checkpoint step 9500.
* Sample size: 100 of 2959 segments.
* Sampling policy: deterministic SHA-256 ranking with seed 20261107; up to
  25 segments carrying a diagnostic marker (negation, quantity/unit,
  name/acronym, pronoun, formal address) and the remainder from the general pool.
* Blinding: the system identity is not shown to the reviewer; outputs are not
  pre-marked as correct.

## Instructions

1. Mark each row OK / MINOR / MAJOR / CRITICAL for meaning preservation.
2. Record error categories (omission, addition, mistranslation, grammar,
   agreement, terminology, punctuation) in the free-text column.
3. If a different wording would also be acceptable, write it in the last column.
   do not require one exact wording.
4. A round-trip translation is **not** evidence of correctness.

Human evaluation status: **pending** until a qualified bilingual reviewer
completes the packet. AI-assisted results are in `REVIEW_RESULTS.md`; they are
not human judgements, and human review remains pending.
