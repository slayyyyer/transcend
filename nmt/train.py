"""Training loop: token-budgeted updates, accumulation, limits, checkpoint/resume."""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .config import Experiment, write_json
from .data.dataset import (
    Example,
    TokenBudgetSampler,
    build_examples,
    chunk_by_length,
    dataset_hash,
    make_batch,
)
from .data.prepare import Pair, read_tsv
from .decode import greedy_decode_batch
from .hardware import peak_rss_bytes, resolve_device, resolve_precision
from .model import INITIALIZATION_MANIFEST, TransformerNMT, build_model
from .seeding import capture_rng_state, restore_rng_state, seed_everything
from .tokenizer import ByteBPETokenizer


@dataclass
class Limits:
    max_steps: int
    max_train_tokens: int
    max_walltime_s: int

    def reached(self, step: int, tokens: int, walltime_s: float) -> str | None:
        if step >= self.max_steps:
            return "max_steps"
        if tokens >= self.max_train_tokens:
            return "max_train_tokens"
        if walltime_s >= self.max_walltime_s:
            return "max_walltime"
        return None


def lr_at(step: int, peak: float, warmup: int) -> float:
    warmup = max(1, warmup)
    if step <= warmup:
        return peak * step / warmup
    return peak * math.sqrt(warmup / step)


def select_pairs(pairs: Sequence[Pair], limit: int, seed: int) -> list[Pair]:
    if limit <= 0 or limit >= len(pairs):
        return list(pairs)
    import hashlib

    ranked = sorted(
        pairs, key=lambda pair: hashlib.sha256(f"{seed}:{pair.pair_id}".encode()).hexdigest()
    )
    return ranked[:limit]


def read_split(directory: Path, split: str) -> list[Pair]:
    path = directory / f"{split}.tsv"
    if not path.exists():
        raise FileNotFoundError(f"missing prepared split: {path}")
    return read_tsv(path)


def _autocast(device: torch.device, precision: str):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return nullcontext()


def compute_validation(
    model: TransformerNMT,
    examples: Sequence[Example],
    tokenizer: ByteBPETokenizer,
    device: torch.device,
    *,
    limit: int,
    max_new_tokens: int,
    beam_size: int,
) -> dict[str, Any]:
    from sacrebleu.metrics import CHRF

    model.eval()
    metric = CHRF(word_order=2)
    report: dict[str, Any] = {"directions": {}, "nll": {}}
    for direction in ("en-fr", "fr-en"):
        subset = [example for example in examples if example.direction == direction][:limit]
        if not subset:
            report["directions"][direction] = {"n": 0, "chrfpp": None}
            continue
        outputs = greedy_decode_batch(
            model,
            [example.src_ids for example in subset],
            max_new_tokens=max_new_tokens,
            eos_id=tokenizer.eos_id,
            pad_id=tokenizer.pad_id,
            device=device,
        )
        hypotheses = [tokenizer.decode(item.ids) for item in outputs]
        references = [tokenizer.decode(example.tgt_out_ids[:-1]) for example in subset]
        score = metric.corpus_score(hypotheses, [references])
        reasons = {}
        for item in outputs:
            reasons[item.termination] = reasons.get(item.termination, 0) + 1

        # Unsmoothed token NLL on the same subset (teacher forcing, no smoothing).
        total_nll = 0.0
        total_tokens = 0
        with torch.no_grad():
            for start in range(0, len(subset), 16):
                batch = make_batch(subset, list(range(start, min(start + 16, len(subset)))))
                logits = model(
                    batch["src_ids"].to(device),
                    batch["src_pad_mask"].to(device),
                    batch["tgt_in_ids"].to(device),
                    batch["tgt_pad_mask"].to(device),
                )
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch["tgt_out_ids"].to(device).reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                total_nll += float(loss)
                total_tokens += int((batch["tgt_out_ids"] != -100).sum())
        nll = total_nll / max(1, total_tokens)
        report["directions"][direction] = {
            "n": len(subset),
            "chrfpp": round(float(score.score), 4),
            "termination": reasons,
            "examples": list(zip(hypotheses[:3], references[:3])),
        }
        report["nll"][direction] = round(nll, 4)
        report["nll"][f"{direction}_perplexity"] = round(math.exp(min(20.0, nll)), 3)
    scores = [v["chrfpp"] for v in report["directions"].values() if v.get("chrfpp") is not None]
    report["mean_chrfpp"] = round(sum(scores) / len(scores), 4) if scores else None
    report["beam_size"] = beam_size
    return report


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def run_training(
    *,
    experiment: Experiment,
    prepared_dir: Path,
    tokenizer: ByteBPETokenizer,
    tokenizer_dir: Path,
    output_dir: Path,
    resume: Path | None = None,
    device_override: str | None = None,
    max_walltime_s: int | None = None,
    max_steps: int | None = None,
    max_train_tokens: int | None = None,
    calibrate: bool = False,
    calibrate_updates: int = 30,
) -> dict[str, Any]:
    seed_everything(experiment.seed, deterministic=True)
    device = resolve_device(device_override or experiment.train.device)
    precision = resolve_precision(experiment.train.precision, device)
    torch.set_num_threads(experiment.train.num_threads)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_pairs = select_pairs(read_split(prepared_dir, "train"), experiment.data.train_pairs, experiment.seed)
    dev_pairs = read_split(prepared_dir, "dev")
    train_examples, train_stats = build_examples(
        train_pairs,
        tokenizer,
        max_source_tokens=experiment.data.max_source_tokens,
        max_target_tokens=experiment.data.max_target_tokens,
    )
    dev_examples, dev_stats = build_examples(
        dev_pairs,
        tokenizer,
        max_source_tokens=experiment.data.max_source_tokens,
        max_target_tokens=experiment.data.max_target_tokens,
    )
    if not train_examples:
        raise RuntimeError("no training examples survived filtering; check token limits")

    batch_size = max(1, experiment.train.max_micro_batch)
    chunks = chunk_by_length(train_examples, batch_size)
    chunk_tokens = [
        sum(train_examples[index].tgt_len for index in chunk) for chunk in chunks
    ]

    model = build_model(experiment.model, experiment.seed).to(device)
    parameter_count = model.parameter_count()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=experiment.train.peak_lr,
        betas=experiment.train.betas,
        eps=experiment.train.eps,
        weight_decay=experiment.train.weight_decay,
    )
    scaler = torch.amp.GradScaler(device.type) if precision == "fp16" else None

    sampler = TokenBudgetSampler(chunk_tokens, seed=experiment.seed)
    global_step = 0
    tokens_seen = 0
    walltime_accumulated = 0.0
    best_metric = -math.inf
    history: list[dict] = []
    validations_without_improvement = 0
    last_validation: dict | None = None

    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = experiment.train.peak_lr
        scaler_state = checkpoint.get("scaler")
        if scaler is not None and scaler_state:
            scaler.load_state_dict(scaler_state)
        global_step = int(checkpoint.get("global_step", 0))
        tokens_seen = int(checkpoint.get("tokens_seen", 0))
        walltime_accumulated = float(checkpoint.get("walltime_s", 0.0))
        best_metric = float(checkpoint.get("best_metric", -math.inf))
        history = list(checkpoint.get("history", []))
        validations_without_improvement = int(checkpoint.get("validations_without_improvement", 0))
        sampler = TokenBudgetSampler.from_state(chunk_tokens, checkpoint.get("sampler", {}))
        restore_rng_state(checkpoint.get("rng"))
        last_validation = checkpoint.get("last_validation")
        print(f"resumed from {resume} at step {global_step}")

    limits = Limits(
        max_steps=max_steps or experiment.train.max_steps,
        max_train_tokens=max_train_tokens or experiment.train.max_train_tokens,
        max_walltime_s=max_walltime_s or experiment.train.max_walltime_s,
    )

    run_manifest = {
        "experiment": experiment.name,
        "seed": experiment.seed,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "precision": precision,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "parameter_count": parameter_count,
        "initialization": INITIALIZATION_MANIFEST,
        "data": {
            "train_pairs_requested": experiment.data.train_pairs,
            "train_pairs_used": len(train_pairs),
            "train_examples": len(train_examples),
            "dev_examples": len(dev_examples),
            "train_stats": train_stats.as_dict(),
            "dev_stats": dev_stats.as_dict(),
            "dataset_hash": dataset_hash(train_examples),
            "source_token_limit": experiment.data.max_source_tokens,
            "target_token_limit": experiment.data.max_target_tokens,
        },
        "tokenizer": {
            "vocab_size": tokenizer.vocab_size,
            "merges": len(tokenizer.merges),
            "sha256": (tokenizer_dir / "merges.sha256").read_text(encoding="utf-8").strip()
            if (tokenizer_dir / "merges.sha256").exists()
            else None,
        },
        "limits": {
            "max_steps": limits.max_steps,
            "max_train_tokens": limits.max_train_tokens,
            "max_walltime_s": limits.max_walltime_s,
            "per_update_target_tokens": experiment.train.token_budget,
        },
        "model_config": json.loads(experiment.to_json())["model"],
        "train_config": json.loads(experiment.to_json())["train"],
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    write_json(output_dir / "resolved_config.json", json.loads(experiment.to_json()))

    def _payload() -> dict:
        return {
            "model": model.state_dict(),
            "config": json.loads(experiment.to_json()),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "global_step": global_step,
            "tokens_seen": tokens_seen,
            "walltime_s": walltime_accumulated + (time.perf_counter() - run_start),
            "sampler": sampler.state_dict(),
            "rng": capture_rng_state(),
            "best_metric": best_metric,
            "history": history,
            "validations_without_improvement": validations_without_improvement,
            "last_validation": last_validation,
            "tokenizer_sha256": run_manifest["tokenizer"]["sha256"],
            "data_hash": run_manifest["data"]["dataset_hash"],
            "device": str(device),
            "precision": precision,
            "seed": experiment.seed,
        }

    model.train()
    tokens_at_start = tokens_seen
    run_start = time.perf_counter()
    group_iterator = sampler.groups(experiment.train.token_budget)
    update_times: list[float] = []
    validation_time = 0.0
    checkpoint_time = 0.0
    stop_reason: str | None = None
    updates_this_run = 0

    while True:
        # --max-walltime is a budget for this invocation. A resumed run must not
        # inherit the previous invocation's elapsed time, or resuming with a
        # fresh budget could stop before doing any work. Cumulative time is
        # still recorded in checkpoints and reports.
        elapsed_this_run = time.perf_counter() - run_start
        stop_reason = limits.reached(global_step, tokens_seen, elapsed_this_run)
        if stop_reason:
            break
        if calibrate and updates_this_run >= calibrate_updates:
            stop_reason = "calibration_complete"
            break

        group_start = time.perf_counter()
        chunk_indices = next(group_iterator)
        optimizer.zero_grad(set_to_none=True)
        total_target_tokens = 0
        loss_sum = 0.0
        padded_source = padded_target = 0

        for chunk_index in chunk_indices:
            indices = chunks[chunk_index]
            batch = make_batch(train_examples, indices)
            total_target_tokens += int(batch["target_tokens"])
            padded_source += int(batch["padded_source_tokens"])
            padded_target += int(batch["padded_target_tokens"])
            src_ids = batch["src_ids"].to(device, non_blocking=True)
            src_mask = batch["src_pad_mask"].to(device, non_blocking=True)
            tgt_in = batch["tgt_in_ids"].to(device, non_blocking=True)
            tgt_mask = batch["tgt_pad_mask"].to(device, non_blocking=True)
            tgt_out = batch["tgt_out_ids"].to(device, non_blocking=True)
            with _autocast(device, precision):
                logits = model(src_ids, src_mask, tgt_in, tgt_mask)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_out.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                    label_smoothing=experiment.train.label_smoothing,
                )
            loss_sum += float(loss.detach())
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if total_target_tokens == 0:
            continue
        # Normalize accumulated gradients by the actual non-padding target tokens,
        # including the last partial accumulation group.
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(total_target_tokens)
        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), experiment.train.clip_norm)
        lr = lr_at(global_step + 1, experiment.train.peak_lr, experiment.train.warmup_updates)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        global_step += 1
        updates_this_run += 1
        tokens_seen += total_target_tokens
        update_seconds = time.perf_counter() - group_start
        update_times.append(update_seconds)

        if global_step % 10 == 0 or global_step == 1:
            print(
                f"step {global_step} loss_sum={loss_sum:.2f} "
                f"loss_per_token={loss_sum / max(1, total_target_tokens):.4f} "
                f"tokens={total_target_tokens} lr={lr:.2e} grad={float(grad_norm):.3f} "
                f"update={update_seconds * 1000:.0f}ms device={device} prec={precision}"
            )

        if calibrate:
            continue

        if global_step % experiment.train.validate_every == 0:
            validation_start = time.perf_counter()
            last_validation = compute_validation(
                model,
                dev_examples,
                tokenizer,
                device,
                limit=experiment.eval.dev_limit,
                max_new_tokens=experiment.eval.max_new_tokens,
                beam_size=experiment.eval.beam_size,
            )
            validation_time += time.perf_counter() - validation_start
            metric = last_validation.get("mean_chrfpp")
            history.append(
                {
                    "step": global_step,
                    "tokens": tokens_seen,
                    "train_loss_per_token": round(loss_sum / max(1, total_target_tokens), 5),
                    "lr": lr,
                    "validation": last_validation,
                }
            )
            write_json(output_dir / "history.json", history)
            write_json(output_dir / "last_validation.json", last_validation)
            if metric is not None and metric > best_metric:
                best_metric = metric
                validations_without_improvement = 0
                _save_checkpoint(output_dir / "best.pt", _payload())
            else:
                validations_without_improvement += 1
            print(
                f"validation step {global_step} mean_chrF++={metric} "
                f"en-fr={last_validation['directions']['en-fr']['chrfpp']} "
                f"fr-en={last_validation['directions']['fr-en']['chrfpp']}"
            )
            if validations_without_improvement >= experiment.train.patience:
                stop_reason = "early_stopping"
                break

        if global_step % experiment.train.checkpoint_every == 0:
            checkpoint_start = time.perf_counter()
            _save_checkpoint(output_dir / "checkpoint.pt", _payload())
            checkpoint_time += time.perf_counter() - checkpoint_start

    elapsed_this_run = time.perf_counter() - run_start
    cumulative_walltime = walltime_accumulated + elapsed_this_run
    tokens_this_run = tokens_seen - tokens_at_start
    if not calibrate:
        _save_checkpoint(output_dir / "checkpoint.pt", _payload())

    recent = update_times[-20:] if update_times else []
    report = {
        "experiment": experiment.name,
        "device": str(device),
        "precision": precision,
        "global_step": global_step,
        "updates_this_run": updates_this_run,
        "tokens_seen": tokens_seen,
        "tokens_this_run": tokens_this_run,
        "walltime_s": round(elapsed_this_run, 2),
        "cumulative_walltime_s": round(cumulative_walltime, 2),
        "mean_update_ms": round(1000 * sum(recent) / len(recent), 2) if recent else None,
        "target_tokens_per_second": round(tokens_this_run / max(1e-9, elapsed_this_run), 2),
        "validation_time_s": round(validation_time, 2),
        "checkpoint_time_s": round(checkpoint_time, 2),
        "stop_reason": stop_reason,
        "best_metric": best_metric if math.isfinite(best_metric) else None,
        "parameter_count": parameter_count,
        "peak_rss_bytes": peak_rss_bytes(),
        "last_validation": last_validation,
        "limits": {
            "max_steps": limits.max_steps,
            "max_train_tokens": limits.max_train_tokens,
            "max_walltime_s": limits.max_walltime_s,
        },
    }
    write_json(output_dir / ("calibration.json" if calibrate else "training_report.json"), report)
    return report
