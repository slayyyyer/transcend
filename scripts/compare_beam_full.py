"""Compare the tuned full newstest2014 beam run with the archived greedy run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median

from sacrebleu.metrics import CHRF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nmt.evaluate import corpus_scores, paired_bootstrap


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def length_stats(hypotheses: list[str]) -> dict:
    lengths = [len(hypothesis.split()) for hypothesis in hypotheses]
    return {
        "mean_words": round(mean(lengths), 3),
        "median_words": median(lengths),
        "short_5_words": sum(length <= 5 for length in lengths),
        "short_8_words": sum(length <= 8 for length in lengths),
        "empty": sum(length == 0 for length in lengths),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--out", default="reports/beam_tuned_step14500.comparison.json")
    parser.add_argument("--markdown", default="reports/beam_tuned_step14500.md")
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    result = {
        "checkpoint": "models/accel_pilot/best.pt",
        "step": 14500,
        "decoder": {"beam_size": 4, "length_penalty": 1.2, "min_new_tokens": 16},
        "baseline": "greedy",
        "directions": {},
    }
    chrf = CHRF(word_order=2, lowercase=False)
    for direction in ("en-fr", "fr-en"):
        greedy = load(reports_dir / f"newstest2014.82m.{direction}.full.json")
        beam = load(reports_dir / f"beam_tuned_step14500.{direction}.full.json")
        greedy_hypotheses = greedy["models"]["candidate"]["hypotheses"]
        beam_hypotheses = beam["models"]["candidate"]["hypotheses"]
        references = greedy["references"]
        doc_ids = greedy.get("doc_ids") or None
        assert len(greedy_hypotheses) == len(beam_hypotheses) == len(references)

        greedy_scores = corpus_scores(greedy_hypotheses, references)
        beam_scores = corpus_scores(beam_hypotheses, references)
        chrf_bootstrap = paired_bootstrap(
            beam_hypotheses,
            greedy_hypotheses,
            references,
            doc_ids,
            samples=args.samples,
            metric="chrfpp",
        )
        bleu_bootstrap = paired_bootstrap(
            beam_hypotheses,
            greedy_hypotheses,
            references,
            doc_ids,
            samples=args.samples,
            metric="bleu",
        )

        wins = losses = ties = 0
        for candidate, baseline, reference in zip(beam_hypotheses, greedy_hypotheses, references):
            candidate_score = chrf.sentence_score(candidate, [reference]).score
            baseline_score = chrf.sentence_score(baseline, [reference]).score
            if candidate_score > baseline_score:
                wins += 1
            elif candidate_score < baseline_score:
                losses += 1
            else:
                ties += 1

        greedy_lengths = [len(hypothesis.split()) for hypothesis in greedy_hypotheses]
        beam_lengths = [len(hypothesis.split()) for hypothesis in beam_hypotheses]
        result["directions"][direction] = {
            "n": len(references),
            "documents": len(set(doc_ids)) if doc_ids else None,
            "greedy": {**greedy_scores, **length_stats(greedy_hypotheses)},
            "beam": {**beam_scores, **length_stats(beam_hypotheses)},
            "delta": {
                "bleu": round(beam_scores["bleu"] - greedy_scores["bleu"], 3),
                "chrfpp": round(beam_scores["chrfpp"] - greedy_scores["chrfpp"], 3),
                "mean_words": round(mean(beam_lengths) - mean(greedy_lengths), 3),
            },
            "paired_bootstrap": {"chrfpp": chrf_bootstrap, "bleu": bleu_bootstrap},
            "sentence_chrfpp": {"beam_wins": wins, "greedy_wins": losses, "ties": ties},
            "changed_outputs": sum(a != b for a, b in zip(beam_hypotheses, greedy_hypotheses)),
            "shorter_outputs": sum(a < b for a, b in zip(beam_lengths, greedy_lengths)),
            "longer_outputs": sum(a > b for a, b in zip(beam_lengths, greedy_lengths)),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Tuned beam search on full newstest2014",
        "",
        "Checkpoint: step 14,500 (`models/accel_pilot/best.pt`). "
        "Beam 4, length penalty 1.2, minimum 16 generated target tokens, selected "
        "on dev only (see `reports/beam_tuning_dev.md`). Greedy is the archived "
        "full-set baseline; both runs use exactly the same retained segments and "
        "references.",
        "",
        "| Direction | Decoder | n | BLEU | chrF++ | mean words | ≤5 words | ≤8 words |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for direction, entry in result["directions"].items():
        for name in ("greedy", "beam"):
            scores = entry[name]
            lines.append(
                f"| {direction} | {name} | {entry['n']} | {scores['bleu']:.3f} | "
                f"{scores['chrfpp']:.3f} | {scores['mean_words']:.3f} | "
                f"{scores['short_5_words']} | {scores['short_8_words']} |"
            )
    lines += [
        "",
        "| Direction | ΔBLEU | ΔchrF++ | paired BLEU Δ [95% CI] | paired chrF++ Δ [95% CI] | sentence chrF++ W/L/T | changed / shorter / longer |",
        "| --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for direction, entry in result["directions"].items():
        chrf_boot = entry["paired_bootstrap"]["chrfpp"]
        bleu_boot = entry["paired_bootstrap"]["bleu"]
        sent = entry["sentence_chrfpp"]
        lines.append(
            f"| {direction} | {entry['delta']['bleu']:+.3f} | {entry['delta']['chrfpp']:+.3f} | "
            f"{bleu_boot['observed_delta']:+.3f} "
            f"[{bleu_boot['ci95'][0]:+.3f}, {bleu_boot['ci95'][1]:+.3f}] | "
            f"{chrf_boot['observed_delta']:+.3f} "
            f"[{chrf_boot['ci95'][0]:+.3f}, {chrf_boot['ci95'][1]:+.3f}] | "
            f"{sent['beam_wins']}/{sent['greedy_wins']}/{sent['ties']} | "
            f"{entry['changed_outputs']}/{entry['shorter_outputs']}/{entry['longer_outputs']} |"
        )
    lines += [
        "",
        "Paired bootstrap resamples the 176 newstest2014 documents (1,000 resamples) "
        "and is one-sided for the p-value shown in the JSON. Sentence W/L/T counts "
        "use sentence-level chrF++; they are descriptive and not a significance test.",
        "",
        "The minimum-length setting prevents EOS before 16 generated target tokens. "
        "It reduces very short outputs but does not remove all mistranslations or "
        "truncation-like omissions; 1,519 EN→FR and 1,365 FR→EN outputs are shorter "
        "than greedy even after the floor. This is an automatic metric comparison, "
        "not a human evaluation; qualified human review remains pending. Greedy "
        "remains the default decoder.",
        "",
    ]
    Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
