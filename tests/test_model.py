import torch

from nmt.config import ModelConfig
from nmt.decode import (
    DecodeOptions,
    SUPPRESSED_IDS,
    beam_search,
    beam_search_batch,
    greedy_decode_batch,
)
from nmt.model import build_model


def tiny_config(**overrides) -> ModelConfig:
    base = dict(vocab_size=300, d_model=64, encoder_layers=2, decoder_layers=2,
                attention_heads=4, ffn_dim=128, dropout=0.1, max_positions=64)
    base.update(overrides)
    return ModelConfig(**base)


def batch(model: ModelConfig, length: int = 7):
    generator = torch.Generator().manual_seed(0)
    src = torch.randint(4, model.vocab_size, (2, length), generator=generator)
    tgt = torch.randint(4, model.vocab_size, (2, length), generator=generator)
    src_mask = torch.zeros_like(src, dtype=torch.bool)
    tgt_mask = torch.zeros_like(tgt, dtype=torch.bool)
    return src, src_mask, tgt, tgt_mask


def test_forward_shapes_and_finite_loss():
    cfg = tiny_config()
    model = build_model(cfg, seed=3)
    src, src_mask, tgt_in, tgt_mask = batch(cfg)
    logits = model(src, src_mask, tgt_in, tgt_mask)
    assert logits.shape == (2, 7, cfg.vocab_size)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, cfg.vocab_size), tgt_in.reshape(-1)
    )
    assert torch.isfinite(loss)


def test_causal_masking_blocks_future_tokens():
    cfg = tiny_config(dropout=0.0)
    model = build_model(cfg, seed=3).eval()
    src, src_mask, tgt_in, tgt_mask = batch(cfg)
    with torch.no_grad():
        first = model(src, src_mask, tgt_in, tgt_mask)
        modified = tgt_in.clone()
        modified[:, -1] = (modified[:, -1] + 1) % cfg.vocab_size
        second = model(src, src_mask, modified, tgt_mask)
    assert torch.allclose(first[:, :-1], second[:, :-1], atol=1e-5)
    assert not torch.allclose(first[:, -1], second[:, -1])


def test_padding_invariance_in_encoder():
    cfg = tiny_config(dropout=0.0)
    model = build_model(cfg, seed=3).eval()
    tokens = torch.randint(4, cfg.vocab_size, (1, 5))
    with torch.no_grad():
        short = model.encode(tokens, torch.zeros_like(tokens, dtype=torch.bool))
        padded = torch.full((1, 9), 0, dtype=torch.long)
        padded[0, :5] = tokens
        mask = torch.ones((1, 9), dtype=torch.bool)
        mask[0, :5] = False
        long = model.encode(padded, mask)
    assert torch.allclose(short, long[:, :5], atol=1e-5)


def test_one_optimizer_step_changes_parameters():
    cfg = tiny_config()
    model = build_model(cfg, seed=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    src, src_mask, tgt_in, tgt_mask = batch(cfg)
    before = model.embed_tokens.weight.detach().clone()
    logits = model(src, src_mask, tgt_in, tgt_mask)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, cfg.vocab_size), tgt_in.reshape(-1))
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    optimizer.step()
    assert not torch.allclose(before, model.embed_tokens.weight.detach())


def test_initialization_is_seeded_and_reproducible():
    cfg = tiny_config()
    first = build_model(cfg, seed=11)
    second = build_model(cfg, seed=11)
    third = build_model(cfg, seed=12)
    for (name_a, a), (name_b, b) in zip(first.named_parameters(), second.named_parameters()):
        assert name_a == name_b
        assert torch.equal(a, b), name_a
    assert not torch.equal(first.embed_tokens.weight, third.embed_tokens.weight)


def test_parameter_count_reported():
    cfg = tiny_config()
    model = build_model(cfg, seed=3)
    counts = model.parameter_count()
    assert counts["total"] == sum(p.numel() for p in model.parameters())
    assert counts["total"] > 0


def test_beam_search_one_matches_greedy_and_beams_are_valid():
    cfg = tiny_config(dropout=0.0)
    model = build_model(cfg, seed=3).eval()
    source = [4, 5, 6, 7, 8]
    greedy = greedy_decode_batch(model, [source], max_new_tokens=8)[0]
    beam_one = beam_search(
        model, source, options=DecodeOptions(max_new_tokens=8, beam_size=1)
    )
    assert beam_one.ids == greedy.ids
    assert beam_one.termination == greedy.termination

    beam = beam_search(
        model, source, options=DecodeOptions(max_new_tokens=8, beam_size=3)
    )
    assert beam.termination in ("eos", "max_length")
    assert all(0 <= token < cfg.vocab_size for token in beam.ids)
    assert not set(beam.ids) & set(SUPPRESSED_IDS)


def test_beam_search_explores_a_better_non_greedy_path():
    class ToyModel:
        def eval(self):
            return self

        def parameters(self):
            return iter([torch.zeros(1)])

        def encode(self, src, mask):
            return torch.zeros(src.shape[0], src.shape[1], 4)

        def decode(self, tgt, memory=None, self_mask=None, cross_mask=None,
                   position_ids=None, past=None, training=False):
            batch = tgt.shape[0]
            logits = torch.full((batch, 1, 300), -20.0)
            for row, token in enumerate(tgt[:, -1].tolist()):
                if token == 259:  # BOS: greedy picks 5, but 6 is available.
                    logits[row, 0, 5] = 0.0
                    logits[row, 0, 6] = -0.1
                elif token == 5:  # Greedy picks 8, but its local mass is split.
                    logits[row, 0, 8] = 0.0
                    logits[row, 0, 9] = 0.0
                    logits[row, 0, 10] = 0.0
                elif token == 6:  # The non-greedy continuation is stronger.
                    logits[row, 0, 7] = 1.0
                elif token == 7:
                    logits[row, 0, 260] = 5.0
                elif token == 8:
                    logits[row, 0, 260] = 0.0
            if past is None:
                past = [{"k": torch.zeros(batch, 1, 1, 1), "v": torch.zeros(batch, 1, 1, 1)}]
            return logits, past

    model = ToyModel()
    source = [4, 5, 6]
    greedy = greedy_decode_batch(model, [source], max_new_tokens=6, device=torch.device("cpu"))[0]
    beam = beam_search(
        model,
        source,
        options=DecodeOptions(max_new_tokens=6, beam_size=4),
        device=torch.device("cpu"),
    )
    assert greedy.ids == [5, 8]
    assert beam.ids == [6, 7]
    assert beam.termination == "eos"


def test_batched_beam_matches_single_sentence_beam():
    cfg = tiny_config(dropout=0.0)
    model = build_model(cfg, seed=3).eval()
    sources = [[4, 5, 6, 7], [8, 9], [10, 11, 12, 13, 14]]
    options = DecodeOptions(max_new_tokens=7, beam_size=3, length_penalty=1.2)
    single = [beam_search(model, source, options=options) for source in sources]
    batched = beam_search_batch(model, sources, options=options)
    assert [item.ids for item in batched] == [item.ids for item in single]
    assert [item.termination for item in batched] == [item.termination for item in single]


def test_min_new_tokens_blocks_early_eos_in_single_and_batched_beam():
    class EosModel:
        cfg = type("Cfg", (), {"vocab_size": 300})()

        def eval(self):
            return self

        def parameters(self):
            return iter([torch.zeros(1)])

        def encode(self, src, mask):
            return torch.zeros(src.shape[0], src.shape[1], 4)

        def decode(self, tgt, memory=None, self_mask=None, cross_mask=None,
                   position_ids=None, past=None, training=False):
            batch = tgt.shape[0]
            logits = torch.full((batch, 1, 300), -20.0)
            logits[:, 0, 260] = 2.0  # EOS
            logits[:, 0, 5] = 1.0
            if past is None:
                past = [{"k": torch.zeros(batch, 1, 1, 1), "v": torch.zeros(batch, 1, 1, 1)}]
            return logits, past

    model = EosModel()
    options = DecodeOptions(max_new_tokens=8, beam_size=2, min_new_tokens=4)
    single = beam_search(model, [4, 5, 6], options=options, device=torch.device("cpu"))
    batched = beam_search_batch(
        model, [[4, 5, 6], [7, 8]], options=options, device=torch.device("cpu")
    )
    assert len(single.ids) >= 4
    assert all(len(item.ids) >= 4 for item in batched)
