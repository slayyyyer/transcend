# AI-assisted bilingual review — 200 examples

**This is not a human evaluation.** The review was performed by the coding agent
(Buffy) to add sentence-level evidence about the step-9,500 model. It does not
replace a qualified bilingual reviewer, and no human judgments are reported.

## Scope and method

* Packets: `reports/review_packet_enfr/review_en-fr.tsv` and
  `reports/review_packet_fren/review_fr-en.tsv`.
* Sample: 100 of 2,959 EN→FR segments and 100 of 2,922 FR→EN segments,
  deterministic SHA-256 ranking with seed 20261107; 25 diagnostic-marker rows
  plus 75 general rows per direction.
* Reviewer inputs: source, reference, system output and diagnostic flags. The
  packet did not display the system identity, but the reviewer knows the project
  and model; this was **not** a blinded human review.
* Rubric:
  * **OK** — meaning preserved; differences are stylistic or equally acceptable.
  * **MINOR** — localized error(s); the main meaning remains recoverable.
  * **MAJOR** — material meaning loss/distortion or broken syntax; unreliable
    without repair.
  * **CRITICAL** — no coherent proposition, unrelated/degenerate output, or a
    critical detail (negation, number, name) destroyed.
  * **Acceptable = OK + MINOR.**
* Error categories may overlap; a row may carry several. Labels were assigned
  against the source and the single reference, without requiring one exact
  wording.

## Results

| Direction | OK | MINOR | MAJOR | CRITICAL | Acceptable (OK+MINOR) |
| --- | ---: | ---: | ---: | ---: | ---: |
| EN→FR | 1 | 14 | 68 | 17 | 15/100 (15.0%) |
| FR→EN | 1 | 17 | 74 | 8 | 18/100 (18.0%) |
| **Both** | **2** | **31** | **142** | **25** | **33/200 (16.5%)** |

Overall, **33/200 outputs were acceptable (16.5%)**: 2 fully OK, 31 MINOR, 142
MAJOR and 25 CRITICAL. In other words, **167/200 (83.5%) were MAJOR or CRITICAL**.
The two directions are broadly similar; EN→FR was slightly worse on this sample
(15% acceptable versus 18%).

## Error categories

| Category | EN→FR rows | FR→EN rows | Total rows | % of 200 |
| --- | ---: | ---: | ---: | ---: |
| mistranslation | 85 | 87 | 172 | 86.0% |
| terminology | 52 | 65 | 117 | 58.5% |
| grammar | 59 | 26 | 85 | 42.5% |
| omission | 36 | 36 | 72 | 36.0% |
| repetition | 24 | 13 | 37 | 18.5% |
| agreement | 4 | 3 | 7 | 3.5% |
| punctuation | 3 | 0 | 3 | 1.5% |
| addition | 0 | 1 | 1 | 0.5% |

Rows can have multiple categories, so the total exceeds 200. The most common
primary category was **mistranslation** (131 rows), followed by **terminology**
(48) and **omission** (11). Grammar and repetition were more prominent in EN→FR;
terminology errors were more prominent in FR→EN.

## Diagnostic-marker slice

| Slice | Rows | Acceptable | Rate |
| --- | ---: | ---: | ---: |
| Marker-bearing | 50 | 3 | 6.0% |
| Unflagged | 150 | 30 | 20.0% |

Marker-bearing rows were deliberately oversampled by the packet design, so this
is a descriptive contrast, not a causal or population estimate.

## Examples

* **OK (EN→FR, id 1182):** "It has always taken place." → "Il a toujours pris
  place." Meaning preserved.
* **OK (FR→EN, id 2761):** "Ça fait partie de la procédure." → "It is part of
  the procedure." Exact.
* **MINOR (EN→FR, id 828):** "However, Yahoo and Google have denied any
  involvement in these practices." → "Mais Yhoo et Google ont déni toute
  implication dans ces pratiques." Meaning is clear; spelling and participle are
  wrong.
* **MINOR (FR→EN, id 254):** "Mais la France a un haut niveau d'impôt et de
  chômage aussi." → "But France has a high level of tax and unemployment." The
  additive "also" is lost, but the proposition survives.
* **MAJOR (EN→FR, id 1688):** "the cost of the new vaccine is likely to be far
  lower" → "le coût de la nouvelle vaccine est probablement loin d'être bas".
  The comparison is reversed: "far lower" becomes "far from low".
* **MAJOR (FR→EN, id 1688):** "Tripodi nie être influencé par Obeid" →
  "Tripodia is influencing by Obeid". The negation is lost, reversing the claim.
* **CRITICAL (EN→FR, id 1120):** "Musicals fill that gap." → "Mainque fasse le
  sap." No coherent proposition survives.
* **CRITICAL (FR→EN, id 2570):** "L'ex-chef de la Gestapo enterré dans un
  cimetière juif" → "The Gapore Chypre Chair is in a judge of judge." Unrelated
  word salad.

## Files

* `reports/review_packet_enfr/review_en-fr_reviewed.tsv` — all 100 EN→FR rows,
  reviewer columns filled.
* `reports/review_packet_fren/review_fr-en_reviewed.tsv` — all 100 FR→EN rows,
  reviewer columns filled.
* `reports/review_summary.json` — machine-readable counts and methodology.
* Original `review_en-fr.tsv` and `review_fr-en.tsv` remain unchanged with empty
  reviewer columns.

## Limitations

* Single AI reviewer; no inter-annotator agreement and no qualified human
  judgments. The labels may be stricter or more lenient than a human reviewer's.
* The packet is a deterministic stratified sample, not a random quality audit.
  Percentages are descriptive, and 100 rows per direction cannot support precise
  subgroup claims.
* Each reference is one human translation; other valid translations were not
  enumerated. Some MINOR labels could reasonably move to OK or MAJOR.
* The review is not a safety, legal, bias, or domain-suitability audit.
* **Qualified bilingual human review remains pending.** It is still the right
  check before making any usability claim.
