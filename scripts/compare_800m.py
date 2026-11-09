#!/usr/bin/env python3
"""Compare continuation checkpoints with a preserved baseline on newstest2014."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean

from sacrebleu.metrics import CHRF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nmt.evaluate import corpus_scores, paired_bootstrap

DECODERS = ("greedy", "beam4")
DIRECTIONS = ("en-fr", "fr-en")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def skipped_count(report: dict) -> int:
    skipped = report.get("skipped_overlength", {})
    if isinstance(skipped, dict):
        return int(skipped.get("n", 0))
    return int(skipped or 0)


def length_stats(hypotheses: list[str]) -> dict:
    lengths = [len(hypothesis.split()) for hypothesis in hypotheses]
    return {
        "mean_words": round(mean(lengths), 3),
        "short_5_words": sum(length <= 5 for length in lengths),
        "short_8_words": sum(length <= 8 for length in lengths),
        "empty": sum(length == 0 for length in lengths),
    }


def compare_decoder(
    report: dict,
    *,
    direction: str,
    baseline_label: str,
    candidate_label: str,
    samples: int,
    chrf: CHRF,
) -> dict:
    if report.get("direction") != direction:
        raise ValueError(
            f"report direction {report.get('direction')!r} != {direction!r}"
        )
    models = report["models"]
    for label in (baseline_label, candidate_label):
        if label not in models:
            raise KeyError(
                f"{direction}: missing model {label!r}; available: {sorted(models)}"
            )

    baseline_model = models[baseline_label]
    candidate_model = models[candidate_label]
    baseline_hypotheses = baseline_model["hypotheses"]
    candidate_hypotheses = candidate_model["hypotheses"]
    references = report["references"]
    doc_ids = report.get("doc_ids") or None
    n = int(report["n"])
    if not (
        len(baseline_hypotheses)
        == len(candidate_hypotheses)
        == len(references)
        == n
    ):
        raise ValueError(
            f"{direction}: length mismatch baseline={len(baseline_hypotheses)} "
            f"candidate={len(candidate_hypotheses)} references={len(references)} n={n}"
        )
    if len(doc_ids or []) not in (0, n):
        raise ValueError(f"{direction}: doc_ids length does not match references")
    if not baseline_hypotheses or not candidate_hypotheses:
        raise ValueError(f"{direction}: empty hypothesis list")

    baseline_scores = corpus_scores(baseline_hypotheses, references)
    candidate_scores = corpus_scores(candidate_hypotheses, references)
    wins = losses = ties = 0
    for candidate, reference, baseline in zip(
        candidate_hypotheses, references, baseline_hypotheses
    ):
        candidate_score = chrf.sentence_score(candidate, [reference]).score
        baseline_score = chrf.sentence_score(baseline, [reference]).score
        if candidate_score > baseline_score:
            wins += 1
        elif candidate_score < baseline_score:
            losses += 1
        else:
            ties += 1

    baseline_lengths = [len(hypothesis.split()) for hypothesis in baseline_hypotheses]
    candidate_lengths = [len(hypothesis.split()) for hypothesis in candidate_hypotheses]
    return {
        "baseline": {
            "label": baseline_label,
            "step": int(baseline_model.get("step", 0)),
            **baseline_scores,
            **length_stats(baseline_hypotheses),
        },
        "candidate": {
            "label": candidate_label,
            "step": int(candidate_model.get("step", 0)),
            **candidate_scores,
            **length_stats(candidate_hypotheses),
        },
        "delta": {
            "bleu": round(candidate_scores["bleu"] - baseline_scores["bleu"], 3),
            "chrfpp": round(candidate_scores["chrfpp"] - baseline_scores["chrfpp"], 3),
        },
        "paired_bootstrap": {
            "chrfpp": paired_bootstrap(
                candidate_hypotheses,
                baseline_hypotheses,
                references,
                doc_ids,
                samples=samples,
                metric="chrfpp",
            ),
            "bleu": paired_bootstrap(
                candidate_hypotheses,
                baseline_hypotheses,
                references,
                doc_ids,
                samples=samples,
                metric="bleu",
            ),
        },
        "sentence_chrfpp": {
            "candidate_wins": wins,
            "baseline_wins": losses,
            "ties": ties,
        },
        "changed_outputs": sum(
            candidate != baseline
            for candidate, baseline in zip(candidate_hypotheses, baseline_hypotheses)
        ),
        "shorter_outputs": sum(
            candidate < baseline
            for candidate, baseline in zip(candidate_lengths, baseline_lengths)
        ),
        "longer_outputs": sum(
            candidate > baseline
            for candidate, baseline in zip(candidate_lengths, baseline_lengths)
        ),
    }


def model_metadata(label: str, reports: dict[tuple[str, str], dict]) -> dict:
    paths = {
        str(report["models"][label]["checkpoint"])
        for report in reports.values()
    }
    steps = {
        int(report["models"][label].get("step", 0))
        for report in reports.values()
    }
    if len(paths) != 1:
        raise ValueError(f"{label}: conflicting checkpoint paths: {sorted(paths)}")
    if len(steps) != 1:
        raise ValueError(f"{label}: conflicting steps: {sorted(steps)}")
    path = Path(paths.pop())
    return {
        "label": label,
        "checkpoint": str(path),
        "step": steps.pop(),
        "sha256": sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--tag", default="scaled_full_800m")
    parser.add_argument("--baseline-label", default="full350")
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=["full800best", "full800final"],
        help="Model labels to compare against the baseline.",
    )
    parser.add_argument("--baseline-step", type=int, default=38000)
    parser.add_argument("--training-report", default="models/scaled_full/training_report.json")
    parser.add_argument(
        "--title",
        default="Full-data 800M continuation vs 350M checkpoint on newstest2014",
    )
    parser.add_argument(
        "--out", default="reports/newstest2014.scaled_full_800m.comparisons.json"
    )
    parser.add_argument(
        "--markdown", default="reports/newstest2014.scaled_full_800m.comparisons.md"
    )
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    chrf = CHRF(word_order=2, lowercase=False)
    reports: dict[tuple[str, str], dict] = {}
    for name in DECODERS:
        for direction in DIRECTIONS:
            path = reports_dir / f"newstest2014.{args.tag}.{name}.{direction}.full.json"
            if not path.exists():
                raise SystemExit(f"missing report: {path}")
            reports[(name, direction)] = load(path)

    candidates = list(dict.fromkeys(args.candidates))
    if args.baseline_label in candidates:
        raise SystemExit("a candidate label may not equal the baseline label")
    labels = [args.baseline_label, *candidates]

    for direction in DIRECTIONS:
        greedy = reports[("greedy", direction)]
        beam = reports[("beam4", direction)]
        if greedy["references"] != beam["references"]:
            raise SystemExit(f"{direction}: greedy and beam-4 references differ")
        if (greedy.get("doc_ids") or []) != (beam.get("doc_ids") or []):
            raise SystemExit(f"{direction}: greedy and beam-4 doc ids differ")
        if greedy["n"] != beam["n"]:
            raise SystemExit(f"{direction}: greedy and beam-4 row counts differ")
        for label in labels:
            if label not in greedy["models"] or label not in beam["models"]:
                raise SystemExit(f"{direction}: missing model {label!r}")
            for report in (greedy, beam):
                if len(report["models"][label]["hypotheses"]) != int(report["n"]):
                    raise SystemExit(f"{direction}: {label} hypothesis count mismatch")

    baseline = model_metadata(args.baseline_label, reports)
    if args.baseline_step is not None and baseline["step"] != args.baseline_step:
        raise SystemExit(
            f"baseline step {baseline['step']} != expected {args.baseline_step}"
        )
    candidate_metadata = {
        label: model_metadata(label, reports) for label in candidates
    }

    training = None
    training_path = Path(args.training_report)
    if training_path.exists():
        report = load(training_path)
        training = {
            key: report.get(key)
            for key in (
                "global_step",
                "tokens_seen",
                "stop_reason",
                "walltime_s",
                "cumulative_walltime_s",
                "best_metric",
            )
        }

    result: dict = {
        "title": args.title,
        "benchmark": "newstest2014 (wmt14/full)",
        "tag": args.tag,
        "training": training,
        "baseline": baseline,
        "candidates": candidate_metadata,
        "decoders": {
            "greedy": {
                "beam_size": int(reports[("greedy", "en-fr")].get("beam_size", 1)),
                "length_penalty": float(
                    reports[("greedy", "en-fr")].get("length_penalty", 1.0)
                ),
                "min_new_tokens": int(
                    reports[("greedy", "en-fr")].get("min_new_tokens", 0)
                ),
            },
            "beam4": {
                "beam_size": int(reports[("beam4", "en-fr")].get("beam_size", 4)),
                "length_penalty": float(
                    reports[("beam4", "en-fr")].get("length_penalty", 1.2)
                ),
                "min_new_tokens": int(
                    reports[("beam4", "en-fr")].get("min_new_tokens", 16)
                ),
            },
        },
        "samples": args.samples,
        "directions": {},
    }

    for direction in DIRECTIONS:
        greedy = reports[("greedy", direction)]
        beam = reports[("beam4", direction)]
        result["directions"][direction] = {
            "n": int(greedy["n"]),
            "documents": greedy.get("documents"),
            "skipped_overlength": skipped_count(greedy),
            "greedy": {
                "baseline": {
                    "label": args.baseline_label,
                    **corpus_scores(
                        greedy["models"][args.baseline_label]["hypotheses"],
                        greedy["references"],
                    ),
                },
                "candidates": {},
            },
            "beam4": {
                "baseline": {
                    "label": args.baseline_label,
                    **corpus_scores(
                        beam["models"][args.baseline_label]["hypotheses"],
                        beam["references"],
                    ),
                },
                "candidates": {},
            },
        }
        result["directions"][direction]["greedy"]["candidates"] = {
            label: compare_decoder(
                greedy,
                direction=direction,
                baseline_label=args.baseline_label,
                candidate_label=label,
                samples=args.samples,
                chrf=chrf,
            )
            for label in candidates
        }
        result["directions"][direction]["beam4"]["candidates"] = {
            label: compare_decoder(
                beam,
                direction=direction,
                baseline_label=args.baseline_label,
                candidate_label=label,
                samples=args.samples,
                chrf=chrf,
            )
            for label in candidates
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        f"# {args.title}",
        "",
        f"Baseline: `{baseline['checkpoint']}` (label `{baseline['label']}`, "
        f"step {baseline['step']:,}; SHA-256 `{baseline['sha256'] or 'unavailable'}`).  ",
    ]
    for label in candidates:
        item = candidate_metadata[label]
        lines.append(
            f"Candidate `{label}`: `{item['checkpoint']}` (step {item['step']:,}; "
            f"SHA-256 `{item['sha256'] or 'unavailable'}`).  "
        )
    if training:
        lines.append(
            "Training report: "
            f"step {training.get('global_step'):,}, "
            f"{training.get('tokens_seen'):,} tokens, "
            f"stop reason `{training.get('stop_reason')}`.  "
        )
    lines += [
        "All checkpoints were evaluated on exactly the same retained newstest2014 "
        "rows with greedy and beam-4 (length penalty 1.2, minimum 16 generated "
        "target tokens). Positive deltas mean the continuation checkpoint is better.",
        "",
        "| Direction | Decoder | Candidate | Candidate step | Baseline BLEU / chrF++ | Candidate BLEU / chrF++ | ΔBLEU | ΔchrF++ | chrF++ 95% CI | Sentence W/L/T |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for direction, entry in result["directions"].items():
        for decoder in DECODERS:
            baseline_scores = entry[decoder]["baseline"]
            for label in candidates:
                item = entry[decoder]["candidates"][label]
                candidate = item["candidate"]
                chrf_boot = item["paired_bootstrap"]["chrfpp"]
                sentence = item["sentence_chrfpp"]
                lines.append(
                    f"| {direction} | {decoder} | {label} | {candidate['step']:,} | "
                    f"{baseline_scores['bleu']:.3f} / {baseline_scores['chrfpp']:.3f} | "
                    f"{candidate['bleu']:.3f} / {candidate['chrfpp']:.3f} | "
                    f"{item['delta']['bleu']:+.3f} | {item['delta']['chrfpp']:+.3f} | "
                    f"[{chrf_boot['ci95'][0]:+.3f}, {chrf_boot['ci95'][1]:+.3f}] | "
                    f"{sentence['candidate_wins']}/{sentence['baseline_wins']}/{sentence['ties']} |"
                )

    same_as_baseline = [
        label
        for label, item in candidate_metadata.items()
        if item["sha256"] is not None and item["sha256"] == baseline["sha256"]
    ]
    if same_as_baseline:
        lines += [
            "",
            "Checkpoint note: "
            + ", ".join(f"`{label}`" for label in same_as_baseline)
            + " is byte-identical to the baseline, so its deltas are exactly zero.",
        ]

    retained = []
    for direction, entry in result["directions"].items():
        total = entry["n"] + entry["skipped_overlength"]
        retained.append(
            f"{entry['n']:,}/{total:,} {direction} "
            f"({entry['skipped_overlength']} over the 128-token source limit)"
        )
    documents = result["directions"]["en-fr"].get("documents") or 0
    lines += [
        "",
        "Retained rows: " + "; ".join(retained) + ".",
        "",
        f"Paired bootstrap resamples the {documents} newstest2014 documents "
        f"({args.samples:,} resamples). Sentence W/L/T uses sentence-level chrF++; "
        "it is descriptive, not a significance test. These are automatic metrics "
        "only, not human evaluation.",
        "",
    ]
    Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown).write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nwrote {args.out} and {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
