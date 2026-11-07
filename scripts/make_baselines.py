"""Create the random-initialization baseline checkpoint and dev benchmarks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nmt.config import experiment_from_dict  # noqa: E402
from nmt.data.prepare import read_tsv  # noqa: E402
from nmt.model import build_model  # noqa: E402
from nmt.tokenizer import ByteBPETokenizer  # noqa: E402


def make_random_checkpoint(config: Path, tokenizer_dir: Path, out_dir: Path) -> None:
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    tokenizer = ByteBPETokenizer.load(tokenizer_dir)
    raw["model"]["vocab_size"] = tokenizer.vocab_size
    experiment = experiment_from_dict(raw, name="random_init")
    model = build_model(experiment.model, experiment.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": json.loads(experiment.to_json()),
            "global_step": 0,
            "tokens_seen": 0,
            "device": "cpu",
            "precision": "fp32",
        },
        out_dir / "checkpoint.pt",
    )
    print(f"random baseline: {model.parameter_count()['total']} parameters -> {out_dir}")


def make_dev_benchmark(prepared: Path, out_dir: Path) -> None:
    pairs = read_tsv(prepared / "dev.tsv")
    out_dir.mkdir(parents=True, exist_ok=True)
    for direction in ("en-fr", "fr-en"):
        rows = []
        for pair in pairs:
            if direction == "en-fr":
                rows.append({"id": pair.pair_id, "docid": f"block-{pair.block}", "src": pair.en, "ref": pair.fr})
            else:
                rows.append({"id": pair.pair_id, "docid": f"block-{pair.block}", "src": pair.fr, "ref": pair.en})
        path = out_dir / f"dev.{direction}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        print(f"{path}: {len(rows)} rows")


if __name__ == "__main__":
    make_random_checkpoint(
        ROOT / "configs" / "existing_accelerator.yaml",
        ROOT / "models" / "cpu_small" / "tokenizer",
        ROOT / "models" / "random_init",
    )
    make_dev_benchmark(ROOT / "data" / "prepared" / "europarl_nc", ROOT / "data" / "prepared" / "europarl_nc")
