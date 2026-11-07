# AI-assisted bilingual review — step 14,500 (200 examples)

**This is not a human evaluation.** The review was performed by the coding agent
(Buffy) to add sentence-level evidence about the step-14,500 checkpoint. It does
not replace a qualified bilingual reviewer, and no human judgments are reported.

## Scope and method

* Model: `models/accel_pilot/best.pt`, optimizer step 14,500 (checkpoint SHA-256
  `cf813c77caed6e7e6785318bfa8630feb70d5f26ec2887f45735a2859c126231`).
* Packets: `reports/review_packet_82m_enfr/review_en-fr.tsv` and
  `reports/review_packet_82m_fren/review_fr-en.tsv`.
* Sample: 100 of 2,959 EN→FR segments and 100 of 2,922 FR→EN segments;
  deterministic SHA-256 ranking with seed 20261107; 25 diagnostic-marker rows
  plus 75 general rows per direction. This is a stratified descriptive sample,
  not a random population audit.
* Reviewer inputs: source, reference, system output and diagnostic flags. The
  packet did not display the system identity, but the reviewer knows the project
  and model; this was **not** a blinded human review.
* Rubric: **OK** — meaning preserved; **MINOR** — localized error(s), main
  meaning recoverable; **MAJOR** — material loss/distortion or broken syntax;
  **CRITICAL** — no coherent proposition, unrelated/degenerate output, or a
  critical detail (negation, number, name) destroyed. **Acceptable = OK + MINOR.**
* Rows can carry multiple error categories. Labels were assigned against the
  source and the single reference, without requiring one exact wording.

## Results

| Direction | OK | MINOR | MAJOR | CRITICAL | Acceptable (OK+MINOR) |
| --- | ---: | ---: | ---: | ---: | ---: |
| EN→FR | 2 | 17 | 53 | 28 | 19/100 (19.0%) |
| FR→EN | 1 | 23 | 54 | 22 | 24/100 (24.0%) |
| **Both** | **3** | **40** | **107** | **50** | **43/200 (21.5%)** |

**43/200 sampled outputs were acceptable (21.5%)**; **157/200 (78.5%) were MAJOR
or CRITICAL**. The two directions are similar, with FR→EN slightly more
acceptable on this sample (24% versus 19%).

## Error categories

| Category | EN→FR rows | FR→EN rows | Total rows |
| --- | ---: | ---: | ---: |
| mistranslation | 88 | 91 | 179 |
| terminology | 48 | 71 | 119 |
| grammar | 73 | 54 | 127 |
| omission | 61 | 61 | 122 |
| repetition | 10 | 9 | 19 |
| agreement | 3 | 1 | 4 |
| addition | 0 | 1 | 1 |

Categories overlap, so the totals exceed 200. Mistranslation, omission, grammar
and terminology are widespread; repetition is less frequent but often co-occurs
with severe degradation.

## Diagnostic-marker slice

| Slice | Rows | Acceptable | Rate |
| --- | ---: | ---: | ---: |
| Marker-bearing | 50 | 8 | 16.0% |
| Unflagged | 150 | 35 | 23.3% |

Marker-bearing rows were deliberately oversampled by the packet design, so this
is a descriptive contrast, not a causal or population estimate. Per direction:
EN→FR 2/25 flagged and 17/75 unflagged; FR→EN 6/25 flagged and 18/75 unflagged.

## Comparison with the archived step-9,500 review

The same deterministic 200 IDs were reviewed for step 9,500 and step 14,500. The
archived review found **33/200 acceptable** (15/100 EN→FR, 18/100 FR→EN); this
review found **43/200** (19/100 EN→FR, 24/100 FR→EN), a net +10 acceptable. Of
the 200 paired rows, **19 became acceptable and 9 stopped being acceptable**;
**26 moved to a less severe label, 41 to a more severe label, and 133 were
unchanged**. The new review also assigns more CRITICAL labels (50 vs 25),
more MINOR labels (40 vs 31), and fewer MAJOR labels (107 vs 142).

That is not a uniform sentence-level improvement. The distribution shifted
toward both more MINOR and more CRITICAL judgments, and this is a fresh
single-AI review rather than an inter-annotator replicate. The corpus
BLEU/chrF++ gains are measured; the sentence-level comparison is descriptive and
does not establish a general quality gain.

## Examples

* **OK (EN→FR, id 1182):** "It has always taken place." → "Il a toujours pris
  place à la place." The meaning survives; the phrasing is slightly awkward.
* **OK (FR→EN, id 2368):** "La banque a rapidement informé les autorités
  compétentes et a coopéré avec les régulateurs." → "The bank quickly informed
  the competent authorities and cooperated with the regulators." Exact.
* **MINOR (EN→FR, id 413):** "I see her eyes resting on me." → "Je vois son eye
  sur le me." The gaze is clear; agreement and pronoun syntax are wrong.
* **MINOR (FR→EN, id 16):** "La technologie est là pour le faire." → "The
  technology is to do so." The availability nuance is weakened, but the
  proposition remains clear.
* **MAJOR (EN→FR, id 1688):** "But the cost of the new vaccine is likely to be far
  lower..." → "Mais le coût du nouveau vaccin est probablement loin d'être bas..."
  The comparison is reversed and the liver/antibody mechanism is lost.
* **MAJOR (FR→EN, id 1727):** "Cela doit inclure la possibilité de durées de
  voyage entre Glasgow et Édimbourg de trois heures au plus." → "This must include
  the possibility of sustainable travel between Glasw and Edimbour and Edimbour
  of three hours." London is omitted and journey time becomes sustainable travel.
* **CRITICAL (EN→FR, id 539):** "New Class Action Claim Against Holy Cross
  Brothers" → "La nouvelle Class Action Cla Holy Reft Walla Fond Frier." The
  proposition is replaced by nonwords.
* **CRITICAL (FR→EN, id 2570):** "L'ex-chef de la Gestapo enterré dans un
  cimetière juif" → "The Gisis Chape of the judge in a judge ciprocal cim of the
  Caporeatical Home Caper." The headline is word salad.

## Files

* `reports/review_packet_82m_enfr/review_en-fr_reviewed.tsv` — all 100 EN→FR rows,
  reviewer columns filled.
* `reports/review_packet_82m_fren/review_fr-en_reviewed.tsv` — all 100 FR→EN rows,
  reviewer columns filled.
* `reports/review_82m_labels.json` — machine-readable row labels, categories and
  notes.
* `reports/review_summary_82m.json` — aggregate counts, methodology and comparison.
* `reports/review_packet_82m_enfr/REVIEW_RESULTS.md` and
  `reports/review_packet_82m_fren/REVIEW_RESULTS.md` — per-direction summaries.

## Limitations

* Single AI reviewer; no qualified human judgments and no inter-annotator
  agreement.
* The 100-per-direction sample cannot support precise subgroup or population
  claims.
* One human reference per row; other acceptable translations were not
  enumerated.
* The reviewer was not blinded to model identity.
* This is not a safety, legal, bias, or domain-suitability audit.
* **Qualified bilingual human review remains pending.** It is still the right
  check before making any usability claim.
