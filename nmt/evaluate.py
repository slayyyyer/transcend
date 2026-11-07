"""Automatic evaluation: SacreBLEU, chrF++, paired bootstrap and baselines."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from sacrebleu.metrics import BLEU, CHRF

from .config import Experiment
from .data.dataset import Example, build_examples, make_batch
from .data.prepare import Pair, read_tsv
from .decode import DecodeOptions, beam_search_batch, greedy_decode_batch
from .model import TransformerNMT
from .tokenizer import ByteBPETokenizer


def _bleu() -> BLEU:
    return BLEU(tokenize="13a", lowercase=False, effective_order=False)


def _chrf() -> CHRF:
    return CHRF(word_order=2, lowercase=False)


def signature(metric) -> str:
    sig = metric.get_signature()
    return sig.format() if hasattr(sig, "format") else str(sig)


def corpus_scores(hypotheses: Sequence[str], references: Sequence[str], indices: Sequence[int] | None = None) -> dict:
    if indices is not None:
        hypotheses = [hypotheses[i] for i in indices]
        references = [references[i] for i in indices]
    bleu = _bleu()
    chrf = _chrf()
    bleu_score = bleu.corpus_score(list(hypotheses), [list(references)])
    chrf_score = chrf.corpus_score(list(hypotheses), [list(references)])
    return {
        "bleu": round(float(bleu_score.score), 3),
        "chrfpp": round(float(chrf_score.score), 3),
        "bleu_signature": signature(bleu),
        "chrfpp_signature": signature(chrf),
        "n": len(hypotheses),
    }


def paired_bootstrap(
    hypotheses_a: Sequence[str],
    hypotheses_b: Sequence[str],
    references: Sequence[str],
    doc_ids: Sequence[str] | None,
    *,
    samples: int = 1000,
    seed: int = 20261107,
    metric: str = "chrfpp",
) -> dict:
    """Paired bootstrap over documents (or segments when no doc ids exist).

    SacreBLEU exposes sentence-level sufficient statistics. Extracting them once
    and summing per resample keeps a full 3,000-segment, 1,000-sample comparison
    practical; rescoring the raw text on every resample took about 4.5 seconds
    per sample on this machine.
    """
    if metric not in ("chrfpp", "bleu"):
        raise ValueError(f"unsupported bootstrap metric: {metric}")
    rng = random.Random(seed)
    units = list(doc_ids) if doc_ids else [str(i) for i in range(len(references))]
    groups: dict[str, list[int]] = {}
    for index, unit in enumerate(units):
        groups.setdefault(unit, []).append(index)
    keys = sorted(groups)
    group_indices = [groups[key] for key in keys]

    metric_fn = _chrf() if metric == "chrfpp" else _bleu()

    def extract(preds: Sequence[str]) -> np.ndarray:
        raw = metric_fn._extract_corpus_statistics(list(preds), [list(references)])
        return np.asarray(raw, dtype=np.int64)

    def aggregate(segment_stats: np.ndarray) -> list[np.ndarray]:
        return [segment_stats[indices].sum(axis=0) for indices in group_indices]

    stats_a = aggregate(extract(hypotheses_a))
    stats_b = aggregate(extract(hypotheses_b))

    def score(document_stats: Sequence[np.ndarray], chosen: Sequence[int]) -> float:
        total = np.sum([document_stats[index] for index in chosen], axis=0)
        return float(metric_fn._compute_score_from_stats(total.tolist()).score)

    all_docs = list(range(len(keys)))
    observed = score(stats_a, all_docs) - score(stats_b, all_docs)
    deltas = []
    for _ in range(max(1, samples)):
        chosen = [rng.randrange(len(keys)) for _ in keys]
        deltas.append(score(stats_a, chosen) - score(stats_b, chosen))
    deltas.sort()
    mean = sum(deltas) / len(deltas)
    lower = deltas[int(0.025 * len(deltas))]
    upper = deltas[int(0.975 * len(deltas)) - 1]
    p_value = sum(1 for delta in deltas if delta <= 0) / len(deltas)
    return {
        "metric": metric,
        "observed_delta": round(observed, 3),
        "bootstrap_mean_delta": round(mean, 3),
        "ci95": [round(lower, 3), round(upper, 3)],
        "p_value_one_sided": round(p_value, 4),
        "samples": len(deltas),
        "resampling_unit": "document" if doc_ids else "segment",
        "unit_count": len(keys),
    }


def _translate_examples(
    model: TransformerNMT,
    tokenizer: ByteBPETokenizer,
    examples: Sequence[Example],
    *,
    max_new_tokens: int,
    beam_size: int,
    device: torch.device,
    length_penalty: float = 1.0,
    min_new_tokens: int = 0,
    batch_size: int = 16,
) -> tuple[list[str], list[str], list[dict]]:
    hypotheses: list[str] = []
    references: list[str] = []
    diagnostics: list[dict] = []
    for start in range(0, len(examples), batch_size):
        subset = examples[start : start + batch_size]
        if beam_size == 1:
            outputs = greedy_decode_batch(
                model,
                [example.src_ids for example in subset],
                max_new_tokens=max_new_tokens,
                eos_id=tokenizer.eos_id,
                pad_id=tokenizer.pad_id,
                device=device,
            )
        else:
            options = DecodeOptions(
                max_new_tokens=max_new_tokens,
                beam_size=beam_size,
                length_penalty=length_penalty,
                min_new_tokens=min_new_tokens,
            )
            outputs = beam_search_batch(
                model,
                [example.src_ids for example in subset],
                options=options,
                device=device,
                pad_id=tokenizer.pad_id,
                eos_id=tokenizer.eos_id,
            )
        for example, decoded in zip(subset, outputs):
            hypotheses.append(tokenizer.decode(decoded.ids))
            references.append(tokenizer.decode(example.tgt_out_ids[:-1]))
            diagnostics.append(
                {
                    "pair_id": example.pair_id,
                    "termination": decoded.termination,
                    "output_tokens": len(decoded.ids),
                }
            )
    return hypotheses, references, diagnostics


def source_copy_baseline(examples: Sequence[Example], tokenizer: ByteBPETokenizer) -> list[str]:
    """Identity baseline: emit the source text as the translation."""
    outputs = []
    for example in examples:
        # Strip the direction control token and EOS, then decode.
        outputs.append(tokenizer.decode(example.src_ids[1:-1]))
    return outputs


@torch.no_grad()
def evaluate_checkpoint(
    experiment: Experiment,
    checkpoint_path: Path,
    tokenizer: ByteBPETokenizer,
    *,
    pairs: Sequence[Pair],
    device: torch.device,
    doc_ids: Sequence[str] | None = None,
    limit: int = 0,
    beam_size: int = 1,
    length_penalty: float = 1.0,
    min_new_tokens: int = 0,
    copies: Sequence[str] | None = None,
    baseline_hypotheses: dict[str, list[str]] | None = None,
    label: str = "candidate",
) -> dict:
    from .model import build_model

    model = build_model(experiment.model, experiment.seed)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()

    result: dict = {
        "label": label,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(payload.get("global_step", 0)),
        "beam_size": beam_size,
        "length_penalty": length_penalty,
        "min_new_tokens": min_new_tokens,
        "directions": {},
    }
    selected_pairs = list(pairs[:limit]) if limit else list(pairs)
    examples, _ = build_examples(
        selected_pairs,
        tokenizer,
        max_source_tokens=experiment.data.max_source_tokens,
        max_target_tokens=experiment.data.max_target_tokens,
    )
    for direction in ("en-fr", "fr-en"):
        subset = [example for example in examples if example.direction == direction]
        if not subset:
            continue
        hypotheses, references, diagnostics = _translate_examples(
            model,
            tokenizer,
            subset,
            max_new_tokens=experiment.eval.max_new_tokens,
            beam_size=beam_size,
            device=device,
            length_penalty=length_penalty,
            min_new_tokens=min_new_tokens,
        )
        scores = corpus_scores(hypotheses, references)
        entry = {
            **scores,
            "hypotheses": hypotheses,
            "references": references,
            "diagnostics": diagnostics,
        }
        if copies is not None:
            copy_hypotheses = source_copy_baseline(subset, tokenizer)
            entry["source_copy"] = corpus_scores(copy_hypotheses, references)
            entry["source_copy_hypotheses"] = copy_hypotheses
        if baseline_hypotheses and direction in baseline_hypotheses:
            baseline = baseline_hypotheses[direction]
            if len(baseline) == len(hypotheses):
                entry["baseline_scores"] = corpus_scores(baseline, references)
                entry["paired_vs_baseline"] = paired_bootstrap(
                    hypotheses, baseline, references, doc_ids
                )
        result["directions"][direction] = entry
    return result


def summarize(result: dict) -> dict:
    directions = result.get("directions", {})
    summary = {"label": result.get("label"), "directions": {}}
    for direction, entry in directions.items():
        summary["directions"][direction] = {
            key: entry[key]
            for key in ("bleu", "chrfpp", "n", "bleu_signature", "chrfpp_signature")
            if key in entry
        }
        if "source_copy" in entry:
            summary["directions"][direction]["source_copy"] = {
                "bleu": entry["source_copy"]["bleu"],
                "chrfpp": entry["source_copy"]["chrfpp"],
            }
        if "baseline_scores" in entry:
            summary["directions"][direction]["baseline"] = {
                "bleu": entry["baseline_scores"]["bleu"],
                "chrfpp": entry["baseline_scores"]["chrfpp"],
            }
        if "paired_vs_baseline" in entry:
            summary["directions"][direction]["paired_vs_baseline"] = entry["paired_vs_baseline"]
    scores = [value["chrfpp"] for value in summary["directions"].values() if "chrfpp" in value]
    summary["mean_chrfpp"] = round(sum(scores) / len(scores), 3) if scores else None
    return summary
