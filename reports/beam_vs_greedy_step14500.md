# Beam search vs greedy — step 14,500

> Historical sample: this is the pre-tuning 495-row comparison. The full
> dev-tuned comparison is in `reports/beam_tuned_step14500.md`.

**Scope:** automatic corpus and sentence-level metric comparison, not a human
evaluation. The checkpoint is `models/accel_pilot/best.pt` at optimizer step
14,500 (SHA-256
`cf813c77caed6e7e6785318bfa8630feb70d5f26ec2887f45735a2859c126231`).

## Method

* Sample: the first 500 raw newstest2014 rows in each direction; 495 rows per
  direction remain after source-length filtering. Each direction covers 22
  documents. This is a deterministic prefix sample, not the full benchmark and
  not a random sample.
* Greedy: existing batch decoder, `beam_size=1`.
* Beam: corrected beam search, `beam_size=4`, `length_penalty=1.0`, with the
  same `max_new_tokens=136`, tokenizer, sources and references.
* Metrics: SacreBLEU `13a` (cased) and chrF++ (`word_order=2`), with paired
  document bootstrap (1,000 resamples over 22 documents). The intervals are
  coarse because only 22 documents are represented.
* No beam width or length penalty was tuned on development data.

## Corpus results

| Direction | Greedy BLEU | Beam-4 BLEU | ΔBLEU [95% CI] | Greedy chrF++ | Beam-4 chrF++ | ΔchrF++ [95% CI] | Changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EN→FR | 10.364 | **13.153** | **+2.789** [2.204, 3.393] | 37.127 | **38.666** | **+1.539** [0.928, 2.119] | 477/495 |
| FR→EN | 10.683 | **13.126** | **+2.443** [1.707, 3.265] | 37.540 | **39.627** | **+2.087** [1.221, 2.801] | 455/495 |

The one-sided bootstrap p-value is reported as `0.000` for both metrics and both
directions (0/1,000 resamples had a non-positive delta). That is not a claim of
full-test significance: the bootstrap has only 22 document units. The simple
direction average is +2.62 BLEU and +1.81 chrF++; this is not a combined corpus
score.

The sample scores are not the full-newstest scores: the earlier full-set run is
11.031/38.667 EN→FR and 11.278/38.855 FR→EN for greedy at step 14,500.

## Sentence-level behavior

| Direction | Improved | Regressed | Tied | Mean sentence ΔchrF++ |
| --- | ---: | ---: | ---: | ---: |
| EN→FR | 301 | 176 | 18 | +1.64 |
| FR→EN | 289 | 165 | 41 | +2.03 |
| **Both** | **590** | **341** | **59** | — |

Across both directions, **932/990 outputs changed** and 58 were identical.
Beam-4 made **341 sentences worse** on sentence chrF++, so the corpus gain is not
a uniform sentence-level improvement.

Beam also shortened many outputs: mean words fell from 24.84 to 22.37 EN→FR and
from 23.51 to 21.51 FR→EN. Beam was shorter on **575/990** rows (equal 186,
longer 229), and produced 25 outputs of at most five words versus 13 for greedy.
The model often assigns high probability to early EOS; length penalty 1.0 does
not prevent those truncations.

## Examples

### EN→FR improvement, id 45

* Reference: *Ce n'est pas quelque chose que les gens veulent.*
* Greedy: *Il ne s'agit pas de personnes qui ont été wanés.*
* Beam-4: *Ce n'est pas quelque chose que les personnes ont été wanés.*

Both outputs are wrong, but beam-4 recovers the reference's opening structure
and the negation; `personnes` is still a mistranslation of `gens`.

### EN→FR regression, id 61

* Reference: *Dans la ville de New York, les responsables des transports
  cherchent à développer un dispositif de taxation qui serait également équipé
  pour payer les frais de stationnement, fournir une assurance « paiement à la
  conduite » et créer un ensemble de données sur la vitesse en temps réel en
  provenance des autres conducteurs que les automobilistes pourraient utiliser
  pour éviter les embouteillages.*
* Greedy (48 words): *En New York City, les fonctionnaires cherchent à
  développer un dévice de vice-verbal qui serait également équipé à payer des
  mènes de mès, fournir des frais de remboursement des frais de remboursement
  des données et de créer une pointe de vendre de données sur le trafic de
  personnaires.*
* Beam-4 (5 words): *En New York City, les*

The beam output stops immediately after the opening phrase. This is the main
failure mode behind the shorter average output.

### FR→EN improvement, id 403

* Reference: *One year previously, she had travelled with my family to Cuba.*
* Greedy: *A parad to the parallel was in the mid-day he was in the case of a
  parallel to Cuba.*
* Beam-4: *A year before, it was in travel with my family to Cuba.*

Beam-4 is substantially more coherent and preserves the travel and family
content; the pronoun/subject is still imperfect.

### FR→EN regression, id 319

* Reference: *Two potholders aged 23 and 27 went missing in a cave under the Dent
  de Crolles on Thursday evening, according to a report from the Isère cave
  rescue organization on Friday.*
* Greedy (30 words): *Twologues of 23 and 27 years were disappeared in a
  gorgorous gap under the Ref of the Reft on Thursday even we have learned to the
  Speo in the Island.*
* Beam-4 (6 words): *Twologues of 23 and 27 years*

Again, beam-4 terminates early and loses the event and location.

## Implementation note

The pre-existing beam implementation did not actually explore: at the first
step it expanded one BOS row into `beam_size` identical rows and ran top-k over
the duplicates, which selected the same token for every beam and collapsed the
search onto greedy. It also dropped completed EOS hypotheses when active beams
outranked them, causing final fallback to a max-length path.

Both issues are fixed in `nmt/decode.py`:

* distinct first tokens are selected from the single BOS row;
* completed hypotheses are retained separately and the best completed sequence
  is returned;
* supported by tests that check beam-1/greedy parity, valid token sequences, and
  a synthetic case where beam must beat a locally greedy path.

`nmt.cli evaluate` now exposes `--beam-size` and `--length-penalty`; the default
remains greedy. At the time of this sample, all **25** Python tests passed; the
current suite has 27.

## Artifacts

* `reports/beam_vs_greedy_step14500.json` — machine-readable metrics, bootstrap
  results, length statistics and examples.
* `reports/beam_vs_greedy_step14500.en-fr.greedy.json` and
  `reports/beam_vs_greedy_step14500.en-fr.beam4.json`.
* `reports/beam_vs_greedy_step14500.fr-en.greedy.json` and
  `reports/beam_vs_greedy_step14500.fr-en.beam4.json`.

## Limitations

* The comparison uses the first 500 raw rows per direction, not the full
  newstest2014 set; it cannot establish full-benchmark deltas.
* Only 22 documents support each bootstrap interval.
* Beam size 4 and length penalty 1.0 were fixed a priori; no beam-width or
  length-penalty sweep was run.
* Corpus metrics do not measure adequacy; the sentence-level chrF++ changes are
  descriptive and include 341 regressions.
* This is not a qualified human evaluation.
