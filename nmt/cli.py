"""Command-line entry points for the whole workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import yaml

from .config import experiment_from_dict, load_experiment, write_json
from .data.build import (
    prepare_fixtures,
    prepare_newstest,
    prepare_training_corpus,
    train_tokenizer,
)
from .data.dataset import Example
from .decode import DecodeOptions, beam_search_batch, greedy_decode_batch
from .evaluate import corpus_scores, paired_bootstrap, summarize
from .export_onnx import export_model
from .hardware import inspect_hardware, resolve_device
from .model import build_model
from .parity import run_parity
from .review_packet import build_review_packet
from .tokenizer import ByteBPETokenizer, BOS_ID
from .train import run_training

REPO_ROOT = Path(__file__).resolve().parents[1]


def _experiment_and_tokenizer(config_path: Path, tokenizer_dir: Path, checkpoint: Path | None = None):
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    tokenizer = ByteBPETokenizer.load(tokenizer_dir)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        raw["model"] = payload["config"]["model"]
    else:
        raw.setdefault("model", {})["vocab_size"] = tokenizer.vocab_size
    return experiment_from_dict(raw, name=str(raw.get("name", "experiment"))), tokenizer


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def translate_texts(
    model,
    tokenizer: ByteBPETokenizer,
    texts: Sequence[str],
    *,
    direction: str,
    max_source_tokens: int,
    max_new_tokens: int,
    device: torch.device,
    beam_size: int = 1,
    length_penalty: float = 1.0,
    min_new_tokens: int = 0,
    batch_size: int = 16,
) -> list[str]:
    source_lang, target_lang = direction.split("-")
    examples: list[Example] = []
    for index, text in enumerate(texts):
        ids = tokenizer.encode(text, lang=target_lang, add_eos=True)
        if len(ids) > max_source_tokens:
            raise ValueError(
                f"input {index} encodes to {len(ids)} tokens, above the model limit of "
                f"{max_source_tokens}; refusing to truncate silently"
            )
        examples.append(
            Example(
                pair_id=str(index),
                direction=direction,
                src_ids=np.asarray(ids, dtype=np.int32),
                tgt_in_ids=np.asarray([BOS_ID], dtype=np.int32),
                tgt_out_ids=np.asarray([tokenizer.eos_id], dtype=np.int32),
            )
        )
    if beam_size > 1:
        options = DecodeOptions(
            max_new_tokens=max_new_tokens,
            beam_size=beam_size,
            length_penalty=length_penalty,
            min_new_tokens=min_new_tokens,
        )
        outputs: list[str] = []
        for start in range(0, len(examples), batch_size):
            subset = examples[start : start + batch_size]
            decoded = beam_search_batch(
                model,
                [item.src_ids for item in subset],
                options=options,
                device=device,
                pad_id=tokenizer.pad_id,
                eos_id=tokenizer.eos_id,
            )
            outputs.extend(tokenizer.decode(item.ids) for item in decoded)
        return outputs
    outputs: list[str] = []
    for start in range(0, len(examples), batch_size):
        subset = examples[start : start + batch_size]
        decoded = greedy_decode_batch(
            model,
            [item.src_ids for item in subset],
            max_new_tokens=max_new_tokens,
            eos_id=tokenizer.eos_id,
            pad_id=tokenizer.pad_id,
            device=device,
        )
        outputs.extend(tokenizer.decode(item.ids) for item in decoded)
    return outputs


def cmd_hardware(args) -> int:
    info = inspect_hardware(REPO_ROOT)
    if args.out:
        write_json(args.out, info)
    print(json.dumps(info, indent=2))
    return 0


def cmd_prepare(args) -> int:
    out = Path(args.out)
    if args.source in ("fixtures", "all"):
        manifest = prepare_fixtures(out / "fixtures")
        print(f"fixtures: {manifest['counts']} splits={manifest['splits']}")
    if args.source in ("europarl_nc", "all"):
        manifest = prepare_training_corpus(
            out / "europarl_nc",
            max_pairs=args.max_pairs,
            block_size=args.block_size,
            max_download_gib=args.max_download_gib,
        )
        print(f"europarl_nc: {manifest['counts']}")
    if args.source in ("newstest2014", "all"):
        manifest = prepare_newstest(out / "newstest2014", max_download_gib=args.max_download_gib)
        print(f"newstest2014: {manifest['segments']} segments, {manifest['documents']} documents")
    return 0


def cmd_tokenizer(args) -> int:
    tokenizer = train_tokenizer(
        Path(args.out),
        train_dir=Path(args.train_dir),
        vocab_size=args.vocab_size,
        min_pair_frequency=args.min_pair_frequency,
        seed=args.seed,
        limits=(args.max_source_tokens, args.max_target_tokens),
    )
    print(f"tokenizer: vocab={tokenizer.vocab_size} merges={len(tokenizer.merges)} -> {args.out}")
    return 0


def cmd_train(args) -> int:
    tokenizer = ByteBPETokenizer.load(Path(args.tokenizer))
    experiment, _ = _experiment_and_tokenizer(Path(args.config), Path(args.tokenizer), Path(args.resume) if args.resume else None)
    report = run_training(
        experiment=experiment,
        prepared_dir=Path(args.prepared),
        tokenizer=tokenizer,
        tokenizer_dir=Path(args.tokenizer),
        output_dir=Path(args.out),
        resume=Path(args.resume) if args.resume else None,
        device_override=args.device,
        max_walltime_s=args.max_walltime,
        max_steps=args.max_steps,
        max_train_tokens=args.max_train_tokens,
        calibrate=args.calibrate,
        calibrate_updates=args.calibrate_updates,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "last_validation"}, indent=2))
    return 0


def cmd_export(args) -> int:
    experiment, tokenizer = _experiment_and_tokenizer(
        Path(args.config), Path(args.tokenizer), Path(args.checkpoint)
    )
    provenance = {}
    if args.provenance:
        provenance = json.loads(Path(args.provenance).read_text(encoding="utf-8"))
    manifest = export_model(
        experiment=experiment,
        checkpoint_path=Path(args.checkpoint),
        tokenizer=tokenizer,
        tokenizer_dir=Path(args.tokenizer),
        output_dir=Path(args.out),
        opset=args.opset,
        provenance=provenance,
    )
    print(f"exported {len(manifest['checksums'])} files to {args.out}")
    return 0


def cmd_parity(args) -> int:
    report = run_parity(Path(args.model_dir), samples=args.samples)
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0 if report["passed"] else 1


def cmd_evaluate(args) -> int:
    rows = _read_jsonl(Path(args.benchmark))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    device = resolve_device(args.device or "cpu")

    # Corpus evaluation drops segments above the model's source limit instead of
    # truncating them; the count is reported so the sample is never implied to
    # be complete.
    target_lang = args.direction.split("-")[1]
    first_label, _, first_path = args.checkpoint[0].partition("=")
    first_experiment, first_tokenizer = _experiment_and_tokenizer(
        Path(args.config), Path(args.tokenizer), Path(first_path or first_label)
    )
    source_limit = first_experiment.data.max_source_tokens
    skipped: list[dict] = []
    kept_rows: list[dict] = []
    for row in rows:
        encoded = first_tokenizer.encode(row["src"], lang=target_lang, add_eos=True)
        if len(encoded) > source_limit:
            skipped.append({"id": row.get("id"), "source_tokens": len(encoded)})
        else:
            kept_rows.append(row)
    rows = kept_rows
    sources = [row["src"] for row in rows]
    references = [row["ref"] for row in rows]
    doc_ids = [row.get("docid", "") for row in rows]

    reports: dict[str, dict] = {}
    hypotheses_by_model: dict[str, list[str]] = {}
    for spec in args.checkpoint:
        label, _, path = spec.partition("=")
        if not path:
            label, path = Path(label).parent.name, label
        experiment, tokenizer = _experiment_and_tokenizer(
            Path(args.config), Path(args.tokenizer), Path(path)
        )
        model = build_model(experiment.model, experiment.seed)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        model.to(device)
        model.eval()
        hypotheses = translate_texts(
            model,
            tokenizer,
            sources,
            direction=args.direction,
            max_source_tokens=experiment.data.max_source_tokens,
            max_new_tokens=args.max_new_tokens or experiment.eval.max_new_tokens,
            device=device,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            min_new_tokens=args.min_new_tokens,
        )
        hypotheses_by_model[label] = hypotheses
        scores = corpus_scores(hypotheses, references)
        reports[label] = {
            "checkpoint": path,
            "step": int(payload.get("global_step", 0)),
            "scores": scores,
            "hypotheses": hypotheses,
        }
        print(f"{label}: BLEU={scores['bleu']} chrF++={scores['chrfpp']}")

    # Identity baseline: emit the source segment unchanged. The earlier version
    # round-tripped through the tokenizer, which measured normalization as well
    # as the copy baseline.
    copy_hypotheses = list(sources)
    copy_scores = corpus_scores(copy_hypotheses, references)

    labels = list(hypotheses_by_model)
    comparisons = {}
    if len(labels) >= 2:
        for label in labels[1:]:
            comparisons[f"{label}_vs_{labels[0]}"] = paired_bootstrap(
                hypotheses_by_model[label],
                hypotheses_by_model[labels[0]],
                references,
                doc_ids if any(doc_ids) else None,
                samples=args.bootstrap,
            )

    result = {
        "benchmark": str(args.benchmark),
        "direction": args.direction,
        "n": len(rows),
        "source_token_limit": source_limit,
        "skipped_overlength": {"n": len(skipped), "details": skipped},
        "documents": len(set(doc_ids)) if any(doc_ids) else None,
        "beam_size": args.beam_size,
        "length_penalty": args.length_penalty,
        "min_new_tokens": args.min_new_tokens,
        "models": reports,
        "source_copy": {"scores": copy_scores, "hypotheses": copy_hypotheses},
        "comparisons": comparisons,
        "sources": sources,
        "references": references,
        "doc_ids": doc_ids,
    }
    summary = {
        "benchmark": str(args.benchmark),
        "direction": args.direction,
        "n": len(rows),
        "skipped_overlength": len(skipped),
        "source_copy": {"bleu": copy_scores["bleu"], "chrfpp": copy_scores["chrfpp"]},
        "models": {label: {**entry["scores"], "step": entry["step"]} for label, entry in reports.items()},
        "beam_size": args.beam_size,
        "length_penalty": args.length_penalty,
        "min_new_tokens": args.min_new_tokens,
        "comparisons": comparisons,
        "signatures": {
            "bleu": copy_scores["bleu_signature"],
            "chrfpp": copy_scores["chrfpp_signature"],
        },
    }
    write_json(args.out, result)
    write_json(str(args.out).replace(".json", "_summary.json"), summary)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_review(args) -> int:
    build_review_packet(
        Path(args.report),
        Path(args.out),
        per_direction=args.per_direction,
        seed=args.seed,
    )
    print(f"review packet written to {args.out}")
    return 0


def cmd_smoke(args) -> int:
    from .smoke import run_smoke

    report = run_smoke(Path(args.out), repo_root=REPO_ROOT)
    print(json.dumps(report, indent=2))
    return 0 if report.get("passed") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nmt", description="From-scratch EN<->FR NMT toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    hardware = sub.add_parser("hardware", help="inspect the actual target hardware")
    hardware.add_argument("--out", default="reports/hardware.json")
    hardware.set_defaults(func=cmd_hardware)

    prepare = sub.add_parser("prepare", help="acquire and prepare corpora")
    prepare.add_argument("--source", default="all", choices=["fixtures", "europarl_nc", "newstest2014", "all"])
    prepare.add_argument("--out", default="data/prepared")
    prepare.add_argument("--max-pairs", type=int, default=300_000)
    prepare.add_argument("--block-size", type=int, default=200)
    prepare.add_argument("--max-download-gib", type=float, default=5.0)
    prepare.set_defaults(func=cmd_prepare)

    tokenizer = sub.add_parser("tokenizer", help="train the joint tokenizer on the train split only")
    tokenizer.add_argument("--train-dir", required=True)
    tokenizer.add_argument("--out", required=True)
    tokenizer.add_argument("--vocab-size", type=int, default=8000)
    tokenizer.add_argument("--min-pair-frequency", type=int, default=2)
    tokenizer.add_argument("--seed", type=int, default=1337)
    tokenizer.add_argument("--max-source-tokens", type=int, default=96)
    tokenizer.add_argument("--max-target-tokens", type=int, default=104)
    tokenizer.set_defaults(func=cmd_tokenizer)

    train = sub.add_parser("train", help="train or resume a model")
    train.add_argument("--config", required=True)
    train.add_argument("--prepared", required=True)
    train.add_argument("--tokenizer", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--resume", default=None)
    train.add_argument("--device", default=None)
    train.add_argument("--max-walltime", type=int, default=None)
    train.add_argument("--max-steps", type=int, default=None)
    train.add_argument("--max-train-tokens", type=int, default=None)
    train.add_argument("--calibrate", action="store_true")
    train.add_argument("--calibrate-updates", type=int, default=30)
    train.set_defaults(func=cmd_train)

    export = sub.add_parser("export", help="export ONNX package")
    export.add_argument("--config", required=True)
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--tokenizer", required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--opset", type=int, default=17)
    export.add_argument("--provenance", default=None)
    export.set_defaults(func=cmd_export)

    parity = sub.add_parser("parity", help="check Python / ONNX Runtime parity")
    parity.add_argument("--model-dir", required=True)
    parity.add_argument("--samples", type=int, default=3)
    parity.add_argument("--out", default=None)
    parity.set_defaults(func=cmd_parity)

    evaluate = sub.add_parser("evaluate", help="score checkpoints on a benchmark")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--tokenizer", required=True)
    evaluate.add_argument("--benchmark", required=True)
    evaluate.add_argument("--direction", required=True, choices=["en-fr", "fr-en"])
    evaluate.add_argument("--checkpoint", action="append", required=True, help="label=path (repeatable)")
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--max-new-tokens", type=int, default=None)
    evaluate.add_argument("--beam-size", type=int, default=1)
    evaluate.add_argument("--length-penalty", type=float, default=1.0)
    evaluate.add_argument("--min-new-tokens", type=int, default=0)
    evaluate.add_argument("--bootstrap", type=int, default=1000)
    evaluate.add_argument("--limit", type=int, default=0, help="score only the first N rows (documented subset)")
    evaluate.set_defaults(func=cmd_evaluate)

    review = sub.add_parser("review-packet", help="build a blind bilingual review packet")
    review.add_argument("--report", required=True)
    review.add_argument("--out", required=True)
    review.add_argument("--per-direction", type=int, default=100)
    review.add_argument("--seed", type=int, default=20261107)
    review.set_defaults(func=cmd_review)

    smoke = sub.add_parser("smoke", help="end-to-end tiny pipeline: tokenizer -> train -> export -> parity")
    smoke.add_argument("--out", default="models/smoke")
    smoke.set_defaults(func=cmd_smoke)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
