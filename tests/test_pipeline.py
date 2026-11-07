from pathlib import Path

import torch

from nmt.data.build import prepare_fixtures
from nmt.data.dataset import TokenBudgetSampler, build_examples, chunk_by_length, make_batch
from nmt.data.prepare import read_tsv
from nmt.tokenizer import ByteBPETokenizer


def prepared(tmp_path: Path) -> Path:
    out = tmp_path / "fixtures"
    prepare_fixtures(out)
    return out


def trained_tokenizer(directory: Path) -> ByteBPETokenizer:
    pairs = read_tsv(directory / "train.tsv")
    texts = [pair.en for pair in pairs] + [pair.fr for pair in pairs]
    return ByteBPETokenizer.train(texts, vocab_size=800, min_pair_frequency=1)


def test_splits_are_disjoint_and_both_directions_inherit_split(tmp_path: Path):
    directory = prepared(tmp_path)
    splits = {name: read_tsv(directory / f"{name}.tsv") for name in ("train", "dev", "test")}
    ids = {name: {pair.pair_id for pair in pairs} for name, pairs in splits.items()}
    assert not (ids["train"] & ids["dev"])
    assert not (ids["train"] & ids["test"])
    assert not (ids["dev"] & ids["test"])
    en = {name: {pair.en for pair in pairs} for name, pairs in splits.items()}
    assert not (en["train"] & en["dev"])
    assert not (en["train"] & en["test"])

    tokenizer = trained_tokenizer(directory)
    examples, stats = build_examples(
        splits["train"], tokenizer, max_source_tokens=64, max_target_tokens=72
    )
    by_pair: dict[str, set[str]] = {}
    for example in examples:
        by_pair.setdefault(example.pair_id, set()).add(example.direction)
    assert all(directions <= {"en-fr", "fr-en"} for directions in by_pair.values())
    assert stats.examples["en-fr"] > 0
    assert stats.examples["fr-en"] > 0


def test_batch_masks_and_shifted_labels(tmp_path: Path):
    directory = prepared(tmp_path)
    tokenizer = trained_tokenizer(directory)
    pairs = read_tsv(directory / "train.tsv")
    examples, _ = build_examples(pairs, tokenizer, max_source_tokens=64, max_target_tokens=72)
    batch = make_batch(examples, [0, 1, 0])
    assert batch["src_ids"].shape[0] == 3
    assert batch["src_pad_mask"].dtype == torch.bool
    assert batch["src_pad_mask"].sum() < batch["src_ids"].numel() or batch["src_ids"].shape[1] == 1
    # Every padded label position must be ignored, and target input is the target
    # output shifted right by one with BOS at the front.
    for row in range(3):
        length = int((batch["tgt_out_ids"][row] != -100).sum())
        assert (batch["tgt_in_ids"][row, 0] == tokenizer.bos_id).item()
        assert torch.equal(
            batch["tgt_in_ids"][row, 1:length], batch["tgt_out_ids"][row, : length - 1]
        )
        assert (batch["tgt_out_ids"][row, length:] == -100).all()


def test_sampler_is_deterministic_and_resumable():
    chunk_tokens = [10, 20, 30, 40, 50, 60]
    first = TokenBudgetSampler(chunk_tokens, seed=7)
    for _ in range(3):
        assert next(first.groups(50))
    second = TokenBudgetSampler.from_state(chunk_tokens, first.state_dict())
    assert next(second.groups(50)) == next(first.groups(50))

    # A state written exactly at an epoch boundary must advance the epoch.
    boundary = TokenBudgetSampler(chunk_tokens, seed=7, epoch=0, chunk_cursor=len(chunk_tokens))
    assert boundary.epoch == 1
    assert boundary.chunk_cursor == 0


def test_chunking_groups_by_length(tmp_path: Path):
    directory = prepared(tmp_path)
    tokenizer = trained_tokenizer(directory)
    pairs = read_tsv(directory / "train.tsv")
    examples, _ = build_examples(pairs, tokenizer, max_source_tokens=64, max_target_tokens=72)
    chunks = chunk_by_length(examples, 8)
    assert sum(len(chunk) for chunk in chunks) == len(examples)
    for chunk in chunks:
        lengths = [examples[index].src_len for index in chunk]
        assert max(lengths) - min(lengths) <= 40
