#!/usr/bin/env python3
"""Compare full-data and scaled step-18,000 newstest2014 reports."""

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

CANDIDATE_LABEL = "scaled_full"
BASELINE_LABEL = "scaled18000"


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


def length_stats(hypotheses: list[str]) -> dict:
    lengths = [len(hypothesis.split()) for hypothesis in hypotheses]
    return {
        "mean_words": round(mean(lengths), 3),
        "short_5_words": sum(length <= 5 for length in lengths),
        "short_8_words": sum(length <= 8 for length in lengths),
        "empty": sum(length == 0 for length in lengths),
    }


def skipped_count(report: dict) -> int:
    skipped = report.get("skipped_overlength", {})
    if isinstance(skipped, dict):
        return int(skipped.get("n", 0))
    return int(skipped or 0)


def compare_decoder(
    report: dict,
    *,
    direction: str,
    samples: int,
    chrf: CHRF,
) -> dict:
    if report.get("direction") != direction:
        raise ValueError(f"report direction {report.get('direction')!r} != {direction!r}")
    models = report["models"]
    if CANDIDATE_LABEL not in models or BASELINE_LABEL not in models:
        raise KeyError(
            f"{direction}: expected labels {CANDIDATE_LABEL!r} and {BASELINE_LABEL!r}; "
            f"available: {sorted(models)}"
        )

    candidate = models[CANDIDATE_LABEL]
    baseline = models[BASELINE_LABEL]
    candidate_hypotheses = candidate["hypotheses"]
    baseline_hypotheses = baseline["hypotheses"]
    references = report["references"]
    doc_ids = report.get("doc_ids") or None
    if not (
        len(candidate_hypotheses)
        == len(baseline_hypotheses)
        == len(references)
        == int(report["n"])
    ):
        raise ValueError(
            f"{direction}: length mismatch candidate={len(candidate_hypotheses)} "
            f"baseline={len(baseline_hypotheses)} references={len(references)} "
            f"n={report['n']}"
        )
    if len(doc_ids or []) not in (0, len(references)):
        raise ValueError(f"{direction}: doc_ids length does not match references")
    if not candidate_hypotheses or not baseline_hypotheses:
        raise ValueError(f"{direction}: empty hypothesis list")

    candidate_scores = corpus_scores(candidate_hypotheses, references)
    baseline_scores = corpus_scores(baseline_hypotheses, references)
    wins = losses = ties = 0
    for hypothesis, reference, previous in zip(
        candidate_hypotheses, references, baseline_hypotheses
    ):
        current_score = chrf.sentence_score(hypothesis, [reference]).score
        previous_score = chrf.sentence_score(previous, [reference]).score
        if current_score > previous_score:
            wins += 1
        elif current_score < previous_score:
            losses += 1
        else:
            ties += 1

    candidate_lengths = [len(hypothesis.split()) for hypothesis in candidate_hypotheses]
    baseline_lengths = [len(hypothesis.split()) for hypothesis in baseline_hypotheses]
    return {
        "n": len(references),
        "documents": len(set(doc_ids)) if doc_ids else None,
        "skipped_overlength": skipped_count(report),
        "candidate": {
            "label": CANDIDATE_LABEL,
            "step": int(candidate.get("step", 0)),
            **candidate_scores,
            **length_stats(candidate_hypotheses),
        },
        "baseline": {
            "label": BASELINE_LABEL,
            "step": int(baseline.get("step", 0)),
            **baseline_scores,
            **length_stats(baseline_hypotheses),
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


def decoder_metadata(report: dict) -> dict:
    return {
        "beam_size": int(report.get("beam_size", 1)),
        "length_penalty": float(report.get("length_penalty", 1.0)),
        "min_new_tokens": int(report.get("min_new_tokens", 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--tag", default="scaled_full")
    parser.add_argument("--candidate", default="models/scaled_full/best.pt")
    parser.add_argument("--baseline", default="models/scaled_pilot/best.pt")
    parser.add_argument(
        "--title", default="Full-data checkpoint vs scaled step 18,000 on newstest2014"
    )
    parser.add_argument(
        "--out", default="reports/newstest2014.scaled_full.comparisons.json"
    )
    parser.add_argument(
        "--markdown", default="reports/newstest2014.scaled_full.comparisons.md"
    )
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    chrf = CHRF(word_order=2, lowercase=False)
    reports: dict[tuple[str, str], dict] = {}
    for name in ("greedy", "beam4"):
        for direction in ("en-fr", "fr-en"):
            path = reports_dir / f"newstest2014.{args.tag}.{name}.{direction}.full.json"
            if not path.exists():
                raise SystemExit(f"missing report: {path}")
            reports[(name, direction)] = load(path)

    candidate_steps = {
        int(report["models"][CANDIDATE_LABEL].get("step", 0))
        for report in reports.values()
    }
    baseline_steps = {
        int(report["models"][BASELINE_LABEL].get("step", 0))
        for report in reports.values()
    }
    if len(candidate_steps) != 1:
        raise SystemExit(f"candidate steps disagree across reports: {sorted(candidate_steps)}")
    if len(baseline_steps) != 1:
        raise SystemExit(f"baseline steps disagree across reports: {sorted(baseline_steps)}")
    candidate_step = candidate_steps.pop()
    baseline_step = baseline_steps.pop()
    if candidate_step <= 0:
        raise SystemExit("candidate checkpoint has no global step")
    if baseline_step != 18000:
        raise SystemExit(f"baseline checkpoint is step {baseline_step}, expected 18000")

    candidate_path = Path(args.candidate)
    baseline_path = Path(args.baseline)
    for report in reports.values():
        reported_candidate = Path(report["models"][CANDIDATE_LABEL]["checkpoint"]).resolve()
        reported_baseline = Path(report["models"][BASELINE_LABEL]["checkpoint"]).resolve()
        if reported_candidate != candidate_path.resolve():
            raise SystemExit(
                f"candidate path in report does not match --candidate: {reported_candidate}"
            )
        if reported_baseline != baseline_path.resolve():
            raise SystemExit(
                f"baseline path in report does not match --baseline: {reported_baseline}"
            )

    for direction in ("en-fr", "fr-en"):
        greedy = reports[("greedy", direction)]
        beam = reports[("beam4", direction)]
        if greedy["references"] != beam["references"]:
            raise SystemExit(f"{direction}: greedy and beam-4 references differ")
        if (greedy.get("doc_ids") or []) != (beam.get("doc_ids") or []):
            raise SystemExit(f"{direction}: greedy and beam-4 doc ids differ")
        if greedy["n"] != beam["n"]:
            raise SystemExit(f"{direction}: greedy and beam-4 row counts differ")

    result: dict = {
        "title": args.title,
        "benchmark": "newstest2014 (wmt14/full)",
        "candidate": {
            "label": CANDIDATE_LABEL,
            "checkpoint": args.candidate,
            "step": candidate_step,
            "sha256": sha256(candidate_path),
        },
        "baseline": {
            "label": BASELINE_LABEL,
            "checkpoint": args.baseline,
            "step": baseline_step,
            "sha256": sha256(baseline_path),
        },
        "decoders": {
            "greedy": decoder_metadata(reports[("greedy", "en-fr")]),
            "beam4": decoder_metadata(reports[("beam4", "en-fr")]),
        },
        "samples": args.samples,
        "directions": {},
    }

    for direction in ("en-fr", "fr-en"):
        result["directions"][direction] = {
            "n": reports[("greedy", direction)]["n"],
            "documents": reports[("greedy", direction)].get("documents"),
            "skipped_overlength": skipped_count(reports[("greedy", direction)]),
            "greedy": compare_decoder(
                reports[("greedy", direction)],
                direction=direction,
                samples=args.samples,
                chrf=chrf,
            ),
            "beam4": compare_decoder(
                reports[("beam4", direction)],
                direction=direction,
                samples=args.samples,
                chrf=chrf,
            ),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        f"# {args.title}",
        "",
        f"Candidate: `{args.candidate}` (step {candidate_step:,}; "
        f"SHA-256 `{result['candidate']['sha256'] or 'unavailable'}`).  ",
        f"Baseline: `{args.baseline}` (step {baseline_step:,}; "
        f"SHA-256 `{result['baseline']['sha256'] or 'unavailable'}`).  ",
        "Both checkpoints were evaluated on exactly the same retained "
        "newstest2014 rows with greedy and beam-4 (length penalty 1.2, "
        "minimum 16 generated target tokens). Positive deltas mean the "
        "full-data checkpoint is better.",
        "",
        "| Direction | Decoder | n | Step 18k BLEU / chrF++ | Full-data BLEU / chrF++ | ΔBLEU | ΔchrF++ | BLEU 95% CI | chrF++ 95% CI | Sentence chrF++ W/L/T |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for direction, entry in result["directions"].items():
        for name in ("greedy", "beam4"):
            item = entry[name]
            base = item["baseline"]
            cand = item["candidate"]
            bleu_boot = item["paired_bootstrap"]["bleu"]
            chrf_boot = item["paired_bootstrap"]["chrfpp"]
            sentence = item["sentence_chrfpp"]
            lines.append(
                f"| {direction} | {name} | {item['n']} | "
                f"{base['bleu']:.3f} / {base['chrfpp']:.3f} | "
                f"{cand['bleu']:.3f} / {cand['chrfpp']:.3f} | "
                f"{item['delta']['bleu']:+.3f} | {item['delta']['chrfpp']:+.3f} | "
                f"[{bleu_boot['ci95'][0]:+.3f}, {bleu_boot['ci95'][1]:+.3f}] | "
                f"[{chrf_boot['ci95'][0]:+.3f}, {chrf_boot['ci95'][1]:+.3f}] | "
                f"{sentence['candidate_wins']}/{sentence['baseline_wins']}/{sentence['ties']} |"
            )

    retained = []
    for direction, entry in result["directions"].items():
        total = entry["n"] + entry["skipped_overlength"]
        retained.append(
            f"{entry['n']:,}/{total:,} {direction} "
            f"({entry['skipped_overlength']} over the 128-token source limit)"
        )
    documents = result["directions"]["en-fr"]["documents"] or 0
    document_word = "document" if documents == 1 else "documents"
    lines += [
        "",
        "Retained rows: " + "; ".join(retained) + ".",
        "",
        f"Paired bootstrap resamples the {documents} newstest2014 {document_word} "
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
