"""End-to-end tiny pipeline: fixtures -> tokenizer -> training -> ONNX -> parity."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from .config import experiment_from_dict, write_json
from .data.build import prepare_fixtures, train_tokenizer
from .export_onnx import export_model
from .parity import run_parity
from .tokenizer import ByteBPETokenizer
from .train import run_training


def run_smoke(out_dir: Path, *, repo_root: Path, max_steps: int = 200) -> dict:
    out_dir = Path(out_dir)
    prepared = out_dir / "prepared"
    tokenizer_dir = out_dir / "tokenizer"
    run_dir = out_dir / "run"

    fixture_manifest = prepare_fixtures(prepared)
    tokenizer = train_tokenizer(
        tokenizer_dir,
        train_dir=prepared,
        vocab_size=2000,
        min_pair_frequency=2,
        seed=1337,
        limits=(64, 72),
    )
    raw = yaml.safe_load((repo_root / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
    raw["model"]["vocab_size"] = tokenizer.vocab_size
    experiment = experiment_from_dict(raw, name="smoke")

    report = run_training(
        experiment=experiment,
        prepared_dir=prepared,
        tokenizer=tokenizer,
        tokenizer_dir=tokenizer_dir,
        output_dir=run_dir,
        max_steps=max_steps,
    )
    checkpoint = run_dir / "checkpoint.pt"
    if not checkpoint.exists():
        checkpoint = run_dir / "best.pt"
    export_dir = out_dir / "export"
    export_model(
        experiment=experiment,
        checkpoint_path=checkpoint,
        tokenizer=tokenizer,
        tokenizer_dir=tokenizer_dir,
        output_dir=export_dir,
        provenance={"source": "fixtures", "purpose": "smoke test"},
    )
    parity = run_parity(export_dir)
    summary = {
        "passed": bool(parity["passed"]),
        "fixtures": fixture_manifest["counts"],
        "tokenizer_vocab": tokenizer.vocab_size,
        "parameters": report["parameter_count"],
        "steps": report["global_step"],
        "tokens_seen": report["tokens_seen"],
        "stop_reason": report["stop_reason"],
        "export": str(export_dir),
        "parity": {
            "worst_max_abs_diff": parity["worst_max_abs_diff"],
            "generation_identical": parity["generation_parity"]["identical"],
        },
    }
    write_json(out_dir / "smoke_report.json", summary)
    return summary
