"""Export the trained model to three ONNX graphs plus a self-describing package.

Graph contract (also recorded in ``model_config.json``):

  encoder(src_ids, src_pad_mask, position_ids) -> memory
  decoder_init(tgt_ids, memory, self_mask, cross_mask, position_ids)
      -> logits, [self_key_i, self_value_i, cross_key_i, cross_value_i]...
  decoder_step(tgt_ids, position_ids, *past) -> logits, *present

All sequence axes are dynamic. Masks are explicit additive float tensors, so no
sequence shape is baked into the graph. Generation always feeds one token to the
decoder, which is why the step graph needs no causal mask.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path
from typing import Sequence

import onnx
import torch

from .config import Experiment, write_json
from .model import INITIALIZATION_MANIFEST, TransformerNMT, build_causal_mask, build_model, build_padding_mask
from .tokenizer import ByteBPETokenizer, EOS_ID, PAD_ID


def _flatten(caches: Sequence[dict]) -> list[torch.Tensor]:
    flat: list[torch.Tensor] = []
    for layer in caches:
        flat.extend([layer["k"], layer["v"], layer["cross_k"], layer["cross_v"]])
    return flat


class EncoderExport(torch.nn.Module):
    def __init__(self, model: TransformerNMT):
        super().__init__()
        self.model = model

    def forward(self, src_ids, src_pad_mask, position_ids):
        return self.model.encode(src_ids, src_pad_mask, position_ids, training=False)


class DecoderInitExport(torch.nn.Module):
    def __init__(self, model: TransformerNMT):
        super().__init__()
        self.model = model

    def forward(self, tgt_ids, memory, self_mask, cross_mask, position_ids):
        logits, caches = self.model.decode(
            tgt_ids,
            memory,
            self_mask=self_mask,
            cross_mask=cross_mask,
            position_ids=position_ids,
            past=None,
            training=False,
        )
        return (logits, *_flatten(caches))


class DecoderStepExport(torch.nn.Module):
    def __init__(self, model: TransformerNMT):
        super().__init__()
        self.model = model

    def forward(self, tgt_ids, position_ids, *past):
        layers = len(self.model.decoder.layers)
        caches = [
            {
                "k": past[4 * index],
                "v": past[4 * index + 1],
                "cross_k": past[4 * index + 2],
                "cross_v": past[4 * index + 3],
            }
            for index in range(layers)
        ]
        logits, new_caches = self.model.decode(
            tgt_ids, None, position_ids=position_ids, past=caches, training=False
        )
        return (logits, *_flatten(new_caches))


def _cache_names(layers: int) -> list[str]:
    names: list[str] = []
    for index in range(layers):
        names.extend(
            [f"self_key_{index}", f"self_value_{index}", f"cross_key_{index}", f"cross_value_{index}"]
        )
    return names


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_axes(name: str, past: bool) -> dict:
    dim = 2
    if name.startswith("self_"):
        return {0: "batch", dim: "past_seq"} if past else {0: "batch", dim: "total_seq"}
    return {0: "batch", dim: "src_seq"}


def export_model(
    *,
    experiment: Experiment,
    checkpoint_path: Path,
    tokenizer: ByteBPETokenizer,
    tokenizer_dir: Path,
    output_dir: Path,
    opset: int = 17,
    provenance: dict | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(experiment.model, experiment.seed)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()

    torch.save(payload["model"], output_dir / "torch_state_dict.pt")
    layers = len(model.decoder.layers)
    cache_names = _cache_names(layers)

    batch, src_len, prefix_len = 2, 7, 2
    src_ids = torch.randint(0, model.cfg.vocab_size, (batch, src_len), dtype=torch.long)
    src_pad = torch.zeros((batch, src_len), dtype=torch.bool)
    src_pos = torch.arange(src_len, dtype=torch.long).unsqueeze(0).expand(batch, src_len)
    tgt_ids = torch.randint(0, model.cfg.vocab_size, (batch, prefix_len), dtype=torch.long)
    tgt_pos = torch.arange(prefix_len, dtype=torch.long).unsqueeze(0).expand(batch, prefix_len)
    memory = model.encode(src_ids, src_pad, src_pos, training=False)
    self_mask = build_causal_mask(prefix_len, prefix_len, 0, src_ids.device, memory.dtype)
    cross_mask = build_padding_mask(src_pad, memory.dtype)
    _, caches = model.decode(
        tgt_ids, memory, self_mask=self_mask, cross_mask=cross_mask, position_ids=tgt_pos
    )
    past = _flatten(caches)

    encoder_path = onnx_dir / "encoder.onnx"
    init_path = onnx_dir / "decoder_init.onnx"
    step_path = onnx_dir / "decoder_step.onnx"

    torch.onnx.export(
        EncoderExport(model),
        (src_ids, src_pad, src_pos),
        encoder_path,
        input_names=["src_ids", "src_pad_mask", "position_ids"],
        output_names=["memory"],
        dynamic_axes={
            "src_ids": {0: "batch", 1: "src_seq"},
            "src_pad_mask": {0: "batch", 1: "src_seq"},
            "position_ids": {0: "batch", 1: "src_seq"},
            "memory": {0: "batch", 1: "src_seq"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    torch.onnx.export(
        DecoderInitExport(model),
        (tgt_ids, memory, self_mask, cross_mask, tgt_pos),
        init_path,
        input_names=["tgt_ids", "memory", "self_mask", "cross_mask", "position_ids"],
        output_names=["logits", *cache_names],
        dynamic_axes={
            "tgt_ids": {0: "batch", 1: "tgt_seq"},
            "memory": {0: "batch", 1: "src_seq"},
            "self_mask": {2: "tgt_seq", 3: "tgt_seq"},
            "cross_mask": {0: "batch", 3: "src_seq"},
            "position_ids": {0: "batch", 1: "tgt_seq"},
            "logits": {0: "batch", 1: "tgt_seq"},
            **{name: _cache_axes(name, past=False) for name in cache_names},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    torch.onnx.export(
        DecoderStepExport(model),
        (tgt_ids[:, :1], tgt_pos[:, :1], *past),
        step_path,
        input_names=["tgt_ids", "position_ids", *[f"past_{name}" for name in cache_names]],
        output_names=["logits", *[f"present_{name}" for name in cache_names]],
        dynamic_axes={
            "tgt_ids": {0: "batch", 1: "tgt_seq"},
            "position_ids": {0: "batch", 1: "tgt_seq"},
            "logits": {0: "batch", 1: "tgt_seq"},
            **{f"past_{name}": _cache_axes(name, past=True) for name in cache_names},
            **{f"present_{name}": _cache_axes(name, past=False) for name in cache_names},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    for path in (encoder_path, init_path, step_path):
        onnx.checker.check_model(str(path))

    tokenizer_out = output_dir / "tokenizer"
    if tokenizer_out.exists():
        shutil.rmtree(tokenizer_out)
    shutil.copytree(tokenizer_dir, tokenizer_out)

    model_config = {
        "architecture": "encoder-decoder Transformer (pre-LN, sinusoidal positions, tied embeddings)",
        "config": json.loads(experiment.to_json())["model"],
        "parameter_count": model.parameter_count(),
        "special_ids": {
            "pad": tokenizer.pad_id,
            "unk": tokenizer.unk_id,
            "bos": tokenizer.bos_id,
            "eos": tokenizer.eos_id,
            "lang_en": tokenizer.lang_id("en"),
            "lang_fr": tokenizer.lang_id("fr"),
            "space": 256,
        },
        "onnx": {
            "files": {
                "encoder": "onnx/encoder.onnx",
                "decoder_init": "onnx/decoder_init.onnx",
                "decoder_step": "onnx/decoder_step.onnx",
            },
            "opset": opset,
            "cache_layout": "K/V tensors are [batch, heads, sequence, head_dim]",
            "cache_order": cache_names,
            "masks": "additive float32; self_mask [1 or B, 1, T, T], cross_mask [B, 1, 1, S]",
            "dtypes": {"src_ids": "int64", "position_ids": "int64", "memory": "float32", "logits": "float32"},
            "dynamic_axes": {"0": "batch", "1": "src_seq or tgt_seq", "2": "seq (cache)"},
        },
        "limits": {
            "max_source_tokens": experiment.data.max_source_tokens,
            "max_target_tokens": experiment.data.max_target_tokens,
            "max_positions": experiment.model.max_positions,
        },
        "checkpoint": {
            "step": int(payload.get("global_step", 0)),
            "tokens_seen": int(payload.get("tokens_seen", 0)),
            "device": payload.get("device"),
            "precision": payload.get("precision"),
        },
        "initialization": INITIALIZATION_MANIFEST,
    }
    write_json(output_dir / "model_config.json", model_config)
    write_json(
        output_dir / "generation.json",
        {
            "decoder_start_token_id": tokenizer.bos_id,
            "eos_token_id": tokenizer.eos_id,
            "pad_token_id": PAD_ID,
            "forced_bos_token": {"en": tokenizer.lang_id("en"), "fr": tokenizer.lang_id("fr")},
            "suppressed_token_ids": [
                PAD_ID,
                tokenizer.bos_id,
                tokenizer.unk_id,
                tokenizer.lang_id("en"),
                tokenizer.lang_id("fr"),
            ],
            "default_max_new_tokens": experiment.eval.max_new_tokens,
            "default_beam_size": experiment.eval.beam_size,
            "length_penalty": experiment.eval.length_penalty,
        },
    )

    # Cross-runtime parity vectors: exact Python greedy output that the Java
    # integration test must reproduce token for token.
    from .decode import greedy_decode_batch

    parity_sentences = [
        ("The train leaves at eight in the morning.", "en", "fr"),
        ("Elle travaille dans une petite librairie près de la rivière.", "fr", "en"),
        ("Nous avons besoin de deux billets pour le concert.", "fr", "en"),
        ("She works in a small bookshop near the river.", "en", "fr"),
    ]
    parity_vectors = []
    parity_budget = min(64, experiment.eval.max_new_tokens)
    for text, source_lang, target_lang in parity_sentences:
        ids = tokenizer.encode(text, lang=target_lang, add_eos=True)
        decoded = greedy_decode_batch(
            model,
            [ids],
            max_new_tokens=parity_budget,
            eos_id=tokenizer.eos_id,
            pad_id=tokenizer.pad_id,
            device=torch.device("cpu"),
        )[0]
        parity_vectors.append(
            {
                "text": text,
                "source": source_lang,
                "target": target_lang,
                "source_ids": ids,
                "expected_ids": decoded.ids,
                "expected_text": tokenizer.decode(decoded.ids),
                "expected_termination": decoded.termination,
                "max_new_tokens": parity_budget,
            }
        )
    write_json(
        output_dir / "parity_vectors.json",
        {"description": "Python greedy output; the Java library must match exactly", "cases": parity_vectors},
    )

    files = sorted(
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "format": "transcend-nmt-onnx-package",
        "version": 1,
        "created_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "experiment": experiment.name,
        "seed": experiment.seed,
        "checksums": {name: _sha256(output_dir / name) for name in files},
        "provenance": provenance or {},
        "no_python_required": True,
        "no_network_required": True,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
