"""Compare the scaled full newstest2014 runs with the final small baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

from sacrebleu.metrics import CHRF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nmt.evaluate import corpus_scores, paired_bootstrap


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def model_entry(report: dict, label: str) -> dict:
    models = report["models"]
    if label not in models:
        raise KeyError(f"missing model {label!r}; available: {sorted(models)}")
    return models[label]


def length_stats(hypotheses: list[str]) -> dict:
    lengths = [len(hypothesis.split()) for hypothesis in hypotheses]
    return {
        "mean_words": round(mean(lengths), 3),
        "short_5_words": sum(length <= 5 for length in lengths),
        "short_8_words": sum(length <= 8 for length in lengths),
        "empty": sum(length == 0 for length in lengths),
    }


def compare(
    direction: str,
    name: str,
    candidate_report: dict,
    candidate_label: str,
    baseline_report: dict,
    baseline_label: str,
    *,
    samples: int,
    chrf: CHRF,
) -> dict:
    candidate = model_entry(candidate_report, candidate_label)["hypotheses"]
    baseline = model_entry(baseline_report, baseline_label)["hypotheses"]
    references = candidate_report["references"]
    doc_ids = candidate_report.get("doc_ids") or None
    if references != baseline_report["references"]:
        raise ValueError(f"{direction}: references differ between reports")
    if len(candidate) != len(baseline) or len(candidate) != len(references):
        raise ValueError(
            f"{direction}: length mismatch candidate={len(candidate)} "
            f"baseline={len(baseline)} references={len(references)}"
        )

    candidate_scores = corpus_scores(candidate, references)
    baseline_scores = corpus_scores(baseline, references)
    wins = losses = ties = 0
    for hypothesis, reference, previous in zip(candidate, references, baseline):
        current_score = chrf.sentence_score(hypothesis, [reference]).score
        previous_score = chrf.sentence_score(previous, [reference]).score
        if current_score > previous_score:
            wins += 1
        elif current_score < previous_score:
            losses += 1
        else:
            ties += 1

    candidate_lengths = [len(hypothesis.split()) for hypothesis in candidate]
    baseline_lengths = [len(hypothesis.split()) for hypothesis in baseline]
    return {
        "comparison": name,
        "n": len(references),
        "documents": len(set(doc_ids)) if doc_ids else None,
        "candidate": {
            "label": candidate_label,
            "step": candidate_report["models"][candidate_label].get("step"),
            **candidate_scores,
            **length_stats(candidate),
        },
        "baseline": {
            "label": baseline_label,
            "step": baseline_report["models"][baseline_label].get("step"),
            **baseline_scores,
            **length_stats(baseline),
        },
        "delta": {
            "bleu": round(candidate_scores["bleu"] - baseline_scores["bleu"], 3),
            "chrfpp": round(candidate_scores["chrfpp"] - baseline_scores["chrfpp"], 3),
        },
        "paired_bootstrap": {
            "chrfpp": paired_bootstrap(
                candidate,
                baseline,
                references,
                doc_ids,
                samples=samples,
                metric="chrfpp",
            ),
            "bleu": paired_bootstrap(
                candidate,
                baseline,
                references,
                doc_ids,
                samples=samples,
                metric="bleu",
            ),
        },
        "sentence_chrfpp": {"candidate_wins": wins, "baseline_wins": losses, "ties": ties},
        "changed_outputs": sum(a != b for a, b in zip(candidate, baseline)),
        "shorter_outputs": sum(a < b for a, b in zip(candidate_lengths, baseline_lengths)),
        "longer_outputs": sum(a > b for a, b in zip(candidate_lengths, baseline_lengths)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--out", default="reports/newstest2014.scaled.comparisons.json")
    parser.add_argument("--markdown", default="reports/newstest2014.scaled.comparisons.md")
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    chrf = CHRF(word_order=2, lowercase=False)
    result: dict = {
        "checkpoint": "models/scaled_pilot/best.pt",
        "decoders": {
            "greedy": {"beam_size": 1, "length_penalty": 1.0, "min_new_tokens": 0},
            "beam4": {"beam_size": 4, "length_penalty": 1.2, "min_new_tokens": 16},
        },
        "baseline": "final step 15,897 small model",
        "directions": {},
    }

    for direction in ("en-fr", "fr-en"):
        scaled_greedy = load(reports_dir / f"newstest2014.scaled.greedy.{direction}.full.json")
        scaled_beam = load(reports_dir / f"newstest2014.scaled.beam4.{direction}.full.json")
        final_greedy = load(reports_dir / f"newstest2014.82m.final.{direction}.full.json")
        final_beam = load(reports_dir / f"newstest2014.82m.final_beam4.{direction}.full.json")

        result["directions"][direction] = {
            "n": scaled_greedy["n"],
            "documents": scaled_greedy.get("documents"),
            "scaled_greedy_vs_final_greedy": compare(
                direction,
                "scaled greedy vs final greedy",
                scaled_greedy,
                "scaled",
                final_greedy,
                "final",
                samples=args.samples,
                chrf=chrf,
            ),
            "scaled_beam4_vs_final_beam4": compare(
                direction,
                "scaled beam-4 vs final beam-4",
                scaled_beam,
                "scaled",
                final_beam,
                "final",
                samples=args.samples,
                chrf=chrf,
            ),
            "scaled_beam4_vs_scaled_greedy": compare(
                direction,
                "scaled beam-4 vs scaled greedy",
                scaled_beam,
                "scaled",
                scaled_greedy,
                "scaled",
                samples=args.samples,
                chrf=chrf,
            ),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Scaled checkpoint on full newstest2014",
        "",
        "Checkpoint: `models/scaled_pilot/best.pt`, selected by the 300-row-per-direction "
        "dev validation. Baseline: the final step-15,897 small model. Greedy and beam-4 "
        "(length penalty 1.2, minimum 16 generated target tokens). Both sides use the same "
        "retained newstest2014 segments and references.",
        "",
        "| Direction | Comparison | n | Baseline BLEU / chrF++ | Scaled BLEU / chrF++ | ΔBLEU | ΔchrF++ | chrF++ 95% CI | Sentence W/L/T | changed / shorter / longer |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for direction, entry in result["directions"].items():
        for key in (
            "scaled_greedy_vs_final_greedy",
            "scaled_beam4_vs_final_beam4",
            "scaled_beam4_vs_scaled_greedy",
        ):
            item = entry[key]
            baseline = item["baseline"]
            candidate = item["candidate"]
            chrf_boot = item["paired_bootstrap"]["chrfpp"]
            sentence = item["sentence_chrfpp"]
            lines.append(
                f"| {direction} | {item['comparison']} | {item['n']} | "
                f"{baseline['bleu']:.3f} / {baseline['chrfpp']:.3f} | "
                f"{candidate['bleu']:.3f} / {candidate['chrfpp']:.3f} | "
                f"{item['delta']['bleu']:+.3f} | {item['delta']['chrfpp']:+.3f} | "
                f"[{chrf_boot['ci95'][0]:+.3f}, {chrf_boot['ci95'][1]:+.3f}] | "
                f"{sentence['candidate_wins']}/{sentence['baseline_wins']}/{sentence['ties']} | "
                f"{item['changed_outputs']}/{item['shorter_outputs']}/{item['longer_outputs']} |"
            )
    lines += [
        "",
        "Paired bootstrap resamples the 176 newstest2014 documents (1,000 resamples). "
        "Sentence W/L/T uses sentence-level chrF++; it is descriptive, not a significance "
        "test. All numbers are automatic metrics, not human evaluation.",
        "",
    ]
    Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
