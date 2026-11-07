"""Tune beam length penalty and minimum length on a deterministic dev sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nmt.cli import _experiment_and_tokenizer
from nmt.cli import translate_texts
from nmt.evaluate import corpus_scores
from nmt.hardware import resolve_device
from nmt.model import build_model


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summary(hypotheses: list[str], references: list[str], greedy: list[str] | None = None) -> dict:
    scores = corpus_scores(hypotheses, references)
    lengths = [len(hypothesis.split()) for hypothesis in hypotheses]
    entry = {
        **scores,
        "mean_words": round(mean(lengths), 3) if lengths else None,
        "median_words": median(lengths) if lengths else None,
        "short_5_words": sum(length <= 5 for length in lengths),
        "short_8_words": sum(length <= 8 for length in lengths),
        "empty": sum(length == 0 for length in lengths),
        "hypotheses": hypotheses,
    }
    if greedy is not None:
        entry["changed_vs_greedy"] = sum(a != b for a, b in zip(hypotheses, greedy))
        entry["shorter_than_greedy"] = sum(
            len(a.split()) < len(b.split()) for a, b in zip(hypotheses, greedy)
        )
    return entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--directions", default="en-fr,fr-en")
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--grid",
        default="[[1.0,0],[1.2,0],[1.5,0],[1.2,8],[1.5,8],[1.5,12]]",
        help="JSON list of [length_penalty, min_new_tokens] pairs",
    )
    args = parser.parse_args()

    grid = [(float(length_penalty), int(min_new_tokens)) for length_penalty, min_new_tokens in json.loads(args.grid)]
    device = resolve_device(args.device)
    experiment, tokenizer = _experiment_and_tokenizer(
        Path(args.config), Path(args.tokenizer), Path(args.checkpoint)
    )
    model = build_model(experiment.model, experiment.seed)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()

    result = {
        "checkpoint": args.checkpoint,
        "step": int(payload.get("global_step", 0)),
        "beam_size": args.beam_size,
        "grid": grid,
        "directions": {},
    }
    for direction in args.directions.split(","):
        target_lang = direction.split("-")[1]
        rows = read_jsonl(Path(f"data/prepared/europarl_nc/dev.{direction}.jsonl"))[::17]
        kept = [
            row
            for row in rows
            if len(tokenizer.encode(row["src"], lang=target_lang, add_eos=True))
            <= experiment.data.max_source_tokens
        ]
        sources = [row["src"] for row in kept]
        references = [row["ref"] for row in kept]
        greedy = translate_texts(
            model,
            tokenizer,
            sources,
            direction=direction,
            max_source_tokens=experiment.data.max_source_tokens,
            max_new_tokens=experiment.eval.max_new_tokens,
            device=device,
            batch_size=args.batch_size,
        )
        entry = {
            "n": len(kept),
            "skipped_overlength": len(rows) - len(kept),
            "sample_stride": 17,
            "greedy": summary(greedy, references),
            "configs": [],
        }
        print(f"{direction}: n={len(kept)} greedy BLEU={entry['greedy']['bleu']} chrF++={entry['greedy']['chrfpp']}", flush=True)
        for length_penalty, min_new_tokens in grid:
            hypotheses = translate_texts(
                model,
                tokenizer,
                sources,
                direction=direction,
                max_source_tokens=experiment.data.max_source_tokens,
                max_new_tokens=experiment.eval.max_new_tokens,
                device=device,
                beam_size=args.beam_size,
                length_penalty=length_penalty,
                min_new_tokens=min_new_tokens,
                batch_size=args.batch_size,
            )
            config = {
                "length_penalty": length_penalty,
                "min_new_tokens": min_new_tokens,
                **summary(hypotheses, references, greedy),
            }
            entry["configs"].append(config)
            print(
                f"{direction} lp={length_penalty} min={min_new_tokens}: "
                f"BLEU={config['bleu']} chrF++={config['chrfpp']} "
                f"mean_words={config['mean_words']} short5={config['short_5_words']} "
                f"changed={config['changed_vs_greedy']} shorter={config['shorter_than_greedy']}",
                flush=True,
            )
        result["directions"][direction] = entry
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
