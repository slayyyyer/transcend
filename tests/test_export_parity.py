import json
from pathlib import Path

import torch

from nmt.config import experiment_from_dict
from nmt.export_onnx import export_model
from nmt.parity import run_parity
from nmt.tokenizer import ByteBPETokenizer

TEXTS = [
    "The train leaves at eight in the morning.",
    "Le train part à huit heures du matin.",
    "Where is the station?",
    "Où se trouve la gare ?",
    "She works in a small bookshop.",
    "Elle travaille dans une petite librairie.",
]


def make_experiment(vocab_size: int):
    return experiment_from_dict(
        {
            "name": "parity",
            "seed": 5,
            "model": {
                "vocab_size": vocab_size,
                "d_model": 32,
                "encoder_layers": 1,
                "decoder_layers": 1,
                "attention_heads": 2,
                "ffn_dim": 64,
                "dropout": 0.0,
                "max_positions": 64,
            },
            "tokenizer": {"target_vocab_size": 400},
            "data": {"source": "fixtures", "max_source_tokens": 32, "max_target_tokens": 32},
            "train": {"device": "cpu", "max_steps": 1, "max_walltime_s": 60},
            "eval": {"max_new_tokens": 12},
        }
    )


def test_export_and_runtime_parity(tmp_path: Path):
    tokenizer = ByteBPETokenizer.train(TEXTS * 5, vocab_size=320, min_pair_frequency=1)
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer.save(tokenizer_dir)
    experiment = make_experiment(tokenizer.vocab_size)

    from nmt.model import build_model

    model = build_model(experiment.model, seed=5)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": json.loads(experiment.to_json()),
            "global_step": 0,
            "tokens_seen": 0,
            "device": "cpu",
            "precision": "fp32",
        },
        checkpoint,
    )
    export_dir = tmp_path / "export"
    manifest = export_model(
        experiment=experiment,
        checkpoint_path=checkpoint,
        tokenizer=tokenizer,
        tokenizer_dir=tokenizer_dir,
        output_dir=export_dir,
    )
    assert "onnx/encoder.onnx" in manifest["checksums"]
    assert "onnx/decoder_init.onnx" in manifest["checksums"]
    assert "onnx/decoder_step.onnx" in manifest["checksums"]
    report = run_parity(export_dir)
    assert report["passed"], report
    assert report["generation_parity"]["identical"]
