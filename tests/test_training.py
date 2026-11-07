from pathlib import Path

import torch

from nmt.config import experiment_from_dict
from nmt.data.build import prepare_fixtures
from nmt.data.prepare import read_tsv
from nmt.decode import greedy_decode_batch
from nmt.model import build_model
from nmt.tokenizer import ByteBPETokenizer
from nmt.train import run_training

PAIRS = [
    ("f001", "Good morning, how are you?", "Bonjour, comment allez-vous ?"),
    ("f002", "I am fine, thank you.", "Je vais bien, merci."),
    ("f003", "My name is Claire.", "Je m'appelle Claire."),
    ("f004", "Where is the train station?", "Où se trouve la gare ?"),
    ("f005", "The train leaves at eight.", "Le train part à huit heures."),
    ("f006", "I would like a coffee, please.", "Je voudrais un café, s'il vous plaît."),
]


def _copy_fixture_subset(tmp_path: Path) -> Path:
    fixture = tmp_path / "fixtures.tsv"
    rows = ["# subset"]
    for pair_id, en, fr in PAIRS:
        rows.append(f"{pair_id}\t{en}\t{fr}")
    fixture.write_text("\n".join(rows) + "\n", encoding="utf-8")
    prepared = tmp_path / "prepared"
    prepare_fixtures(prepared, fixture_path=fixture)
    return prepared


def _experiment(vocab_size: int, steps: int = 400):
    return experiment_from_dict(
        {
            "name": "memorization",
            "seed": 21,
            "model": {
                "vocab_size": vocab_size,
                "d_model": 64,
                "encoder_layers": 1,
                "decoder_layers": 1,
                "attention_heads": 2,
                "ffn_dim": 128,
                "dropout": 0.0,
                "max_positions": 128,
            },
            "tokenizer": {"target_vocab_size": 600},
            "data": {"source": "fixtures", "max_source_tokens": 64, "max_target_tokens": 64},
            "train": {
                "device": "cpu",
                "precision": "fp32",
                "num_threads": 2,
                "peak_lr": 3e-3,
                "warmup_updates": 20,
                "token_budget": 128,
                "max_micro_batch": 16,
                "max_steps": steps,
                "max_train_tokens": 10_000_000,
                "max_walltime_s": 300,
                "checkpoint_every": 10000,
                "validate_every": 10000,
                "patience": 100,
                "label_smoothing": 0.0,
            },
            "eval": {"max_new_tokens": 64, "dev_limit": 8},
        }
    )


def test_tiny_corpus_memorization(tmp_path: Path):
    prepared = _copy_fixture_subset(tmp_path)
    pairs = read_tsv(prepared / "train.tsv")
    tokenizer = ByteBPETokenizer.train(
        [p.en for p in pairs] + [p.fr for p in pairs], vocab_size=600, min_pair_frequency=1
    )
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer.save(tokenizer_dir)
    experiment = _experiment(tokenizer.vocab_size)

    run_dir = tmp_path / "run"
    report = run_training(
        experiment=experiment,
        prepared_dir=prepared,
        tokenizer=tokenizer,
        tokenizer_dir=tokenizer_dir,
        output_dir=run_dir,
    )
    assert report["global_step"] == 400
    payload = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    model = build_model(experiment.model, experiment.seed)
    model.load_state_dict(payload["model"])
    model.eval()

    for direction in ("en-fr", "fr-en"):
        source_lang, target_lang = direction.split("-")
        for pair in pairs:
            source = pair.en if direction == "en-fr" else pair.fr
            expected = pair.fr if direction == "en-fr" else pair.en
            ids = tokenizer.encode(source, lang=target_lang, add_eos=True)
            decoded = greedy_decode_batch(
                model,
                [ids],
                max_new_tokens=64,
                eos_id=tokenizer.eos_id,
                pad_id=tokenizer.pad_id,
            )[0]
            assert decoded.termination == "eos"
            assert tokenizer.decode(decoded.ids) == expected, (direction, source, tokenizer.decode(decoded.ids), expected)


def test_resume_matches_uninterrupted_cpu_run(tmp_path: Path):
    prepared = _copy_fixture_subset(tmp_path)
    pairs = read_tsv(prepared / "train.tsv")
    tokenizer = ByteBPETokenizer.train(
        [p.en for p in pairs] + [p.fr for p in pairs], vocab_size=600, min_pair_frequency=1
    )
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer.save(tokenizer_dir)

    experiment = _experiment(tokenizer.vocab_size, steps=30)

    straight = tmp_path / "straight"
    run_training(
        experiment=experiment,
        prepared_dir=prepared,
        tokenizer=tokenizer,
        tokenizer_dir=tokenizer_dir,
        output_dir=straight,
    )

    staged = tmp_path / "staged"
    run_training(
        experiment=experiment,
        prepared_dir=prepared,
        tokenizer=tokenizer,
        tokenizer_dir=tokenizer_dir,
        output_dir=staged,
        max_steps=15,
    )
    resumed = run_training(
        experiment=experiment,
        prepared_dir=prepared,
        tokenizer=tokenizer,
        tokenizer_dir=tokenizer_dir,
        output_dir=staged,
        resume=staged / "checkpoint.pt",
        max_steps=30,
    )
    assert resumed["global_step"] == 30

    straight_state = torch.load(straight / "checkpoint.pt", map_location="cpu", weights_only=False)["model"]
    staged_state = torch.load(staged / "checkpoint.pt", map_location="cpu", weights_only=False)["model"]
    for key in straight_state:
        assert torch.equal(straight_state[key], staged_state[key]), key


def test_resume_gets_a_fresh_walltime_budget(tmp_path: Path):
    """A resumed invocation must not inherit the prior run's wall-clock usage."""
    prepared = _copy_fixture_subset(tmp_path)
    pairs = read_tsv(prepared / "train.tsv")
    tokenizer = ByteBPETokenizer.train(
        [p.en for p in pairs] + [p.fr for p in pairs], vocab_size=600, min_pair_frequency=1
    )
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer.save(tokenizer_dir)
    experiment = _experiment(tokenizer.vocab_size, steps=10)

    run_dir = tmp_path / "run"
    run_training(
        experiment=experiment,
        prepared_dir=prepared,
        tokenizer=tokenizer,
        tokenizer_dir=tokenizer_dir,
        output_dir=run_dir,
        max_steps=5,
        max_walltime_s=60,
    )

    checkpoint_path = run_dir / "checkpoint.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["walltime_s"] = 9999.0
    torch.save(payload, checkpoint_path)

    resumed = run_training(
        experiment=experiment,
        prepared_dir=prepared,
        tokenizer=tokenizer,
        tokenizer_dir=tokenizer_dir,
        output_dir=run_dir,
        resume=checkpoint_path,
        max_steps=10,
        max_walltime_s=60,
    )
    assert resumed["global_step"] == 10
    assert resumed["walltime_s"] < 60
    assert resumed["cumulative_walltime_s"] >= 9999.0
    assert resumed["tokens_this_run"] > 0
