# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validation gates for the gemma4_31b vision-tower quantization recipe.

Two test classes:

* ``TestVisionQuantRoundtrip`` — Gate 2. Loads HF Gemma4VisionModel as bf16,
  loads the same weights into our ported tower, applies the new
  ``quantize_vision_position_table`` recipe (PE table → INT8 per-channel,
  every other vision tensor stays bf16), and verifies cosine_sim > 0.999
  versus the HF reference output on a small fixed input.

* ``TestBackwardCompatTextOnly`` — Gate 1 (regression) + Gate 3
  (backward-compat read). On a tiny model:
    1. Quantize + save with NO vision attached → safetensors keys must
       equal the pre-port _text_only_state_dict_keys.json snapshot
       (after the ``embed_tokens.weight`` torchao split).
    2. Quantize + save with vision DETACHED in quantize_and_save (the
       --no-vision code path) on a model that has vision attached must
       produce the same on-disk key set.
    3. ``load_prequantized_model`` on a text-only safetensors must
       reconstruct a model whose state_dict keys match the text-only
       snapshot (i.e. it does NOT attach a vision_tower submodule when
       the file has no vision keys, and does not error).

These tests are CPU-only except for the cosine-sim check, which loads the
HF reference (large) and is gated on ``GEMMA4_31B_HF_DIR`` existing.
"""

from __future__ import annotations

import json
import os
import re as _re
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Make the repo importable as `executorch.*` regardless of which conda picks up.
_REPO_ROOT = "/home/gasoonjia/executorch"
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from executorch.examples.models.gemma4_31b.model import (  # noqa: E402
    Gemma4_31B,
    Gemma4_31BConfig,
)
from executorch.examples.models.gemma4_31b.quant.pack_vision_cuda import (  # noqa: E402
    collect_vision_state_dict,
    has_vision_keys,
    install_int8_pe_dispatch,
    quantize_vision_position_table,
)
from executorch.examples.models.gemma4_31b.vision_tower import (  # noqa: E402
    Gemma4_31BVisionTower,
    Gemma4VisionConfig,
    hf_vision_key_map,
    hf_vision_per_layer_key_map,
)
from safetensors.torch import save_file  # noqa: E402
from torchao.prototype.safetensors.safetensors_support import (  # noqa: E402
    flatten_tensor_state_dict,
)


HF_MODEL_DIR = "/home/gasoonjia/models/gemma-4-31B"
SNAPSHOT_PATH = (
    Path(__file__).parent.parent.parent / "tests" / "_text_only_state_dict_keys.json"
)


def _expected_text_only_keys() -> set[str]:
    with open(SNAPSHOT_PATH, "r") as f:
        return set(json.load(f))


# ---------------------------------------------------------------------------
# Test 1 — vision_tower with INT8 PE table matches HF bf16 reference.
# ---------------------------------------------------------------------------


def _hf_to_our_vision_state_dict(hf_state: dict) -> dict:
    """Same key remap as tests/test_vision_tower.py (kept private here to
    avoid cross-test imports)."""
    fixed = hf_vision_key_map()
    per_layer = hf_vision_per_layer_key_map()
    out: dict = {}
    for k, v in hf_state.items():
        norm = k
        if not (
            norm.startswith("model.vision_tower.")
            or norm.startswith("model.embed_vision.")
        ):
            norm = "model." + norm if not norm.startswith("model.") else norm
        if norm in fixed:
            out[fixed[norm]] = v
            continue
        for hf_pat, model_pat in per_layer.items():
            regex = _re.escape(hf_pat).replace(r"\{\}", r"(\d+)")
            m = _re.fullmatch(regex, norm)
            if m:
                out[model_pat.replace("{}", m.group(1), 1)] = v
                break
    return out


def _build_hf_vision_wrapper(dtype: torch.dtype):
    """Load just HF's vision_tower + embed_vision."""
    from transformers import Gemma4ForConditionalGeneration

    model = Gemma4ForConditionalGeneration.from_pretrained(
        HF_MODEL_DIR,
        dtype=dtype,
        device_map="cpu",
    )
    vision_tower = model.model.vision_tower
    embed_vision = model.model.embed_vision

    class _HFVisionWrapper(torch.nn.Module):
        def __init__(self, vt, ev):
            super().__init__()
            self.patch_embedder = vt.patch_embedder
            self.encoder = vt.encoder
            self.pooler = vt.pooler
            self.embed_vision = ev
            self.standardize = bool(getattr(vt.config, "standardize", False))
            if self.standardize:
                self.std_bias = vt.std_bias
                self.std_scale = vt.std_scale
            self.pooling_kernel_size = vt.config.pooling_kernel_size

        def forward(self, pixel_values, pixel_position_ids):
            pks = self.pooling_kernel_size
            output_length = pixel_values.shape[1] // (pks * pks)
            padding_positions = (pixel_position_ids == -1).all(dim=-1)
            inputs_embeds = self.patch_embedder(
                pixel_values, pixel_position_ids, padding_positions
            )
            encoder_output = self.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=~padding_positions,
                pixel_position_ids=pixel_position_ids,
            )
            hidden_states, pooler_mask = self.pooler(
                hidden_states=encoder_output.last_hidden_state,
                pixel_position_ids=pixel_position_ids,
                padding_positions=padding_positions,
                output_length=output_length,
            )
            if self.standardize:
                hidden_states = (hidden_states - self.std_bias) * self.std_scale
                hidden_states = hidden_states.masked_fill(
                    ~pooler_mask.unsqueeze(-1), 0.0
                )
            return self.embed_vision(hidden_states), pooler_mask

    wrapper = _HFVisionWrapper(vision_tower, embed_vision).eval()
    text_hidden = model.config.text_config.hidden_size

    hf_state = {}
    for k, v in vision_tower.state_dict().items():
        hf_state["model.vision_tower." + k] = v
    for k, v in embed_vision.state_dict().items():
        hf_state["model.embed_vision." + k] = v

    del model
    import gc as _gc

    _gc.collect()
    return wrapper, hf_state, text_hidden


def _make_inputs(batch: int, grid: int, patch_dim: int, dtype: torch.dtype):
    g = torch.Generator().manual_seed(0)
    num_patches = grid * grid
    pixel_values = torch.rand(batch, num_patches, patch_dim, generator=g, dtype=dtype)
    coords = torch.arange(grid, dtype=torch.long)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    pos = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    pixel_position_ids = pos.unsqueeze(0).expand(batch, -1, -1).contiguous()
    return pixel_values, pixel_position_ids


@pytest.mark.skipif(
    not os.path.exists(HF_MODEL_DIR), reason="HF Gemma 4 31B checkpoint not present"
)
def test_vision_quant_roundtrip():
    """Cosine sim > 0.999 between HF eager bf16 and our packed
    (PE-int8 + linears bf16) tower on a small fixed input."""
    torch.manual_seed(0)
    dtype = torch.bfloat16

    cfg = Gemma4VisionConfig.from_hf_config(os.path.join(HF_MODEL_DIR, "config.json"))
    pixel_values, pixel_position_ids = _make_inputs(
        batch=1, grid=6, patch_dim=cfg.patch_dim, dtype=dtype
    )

    # ---- HF reference (bf16) ----
    hf_wrapper, hf_state, text_hidden = _build_hf_vision_wrapper(dtype)
    with torch.no_grad():
        hf_emb, hf_mask = hf_wrapper(pixel_values, pixel_position_ids)

    # ---- Our ported tower w/ INT8 PE table ----
    our_tower = (
        Gemma4_31BVisionTower(cfg, text_hidden_size=text_hidden).to(dtype).eval()
    )
    our_state = _hf_to_our_vision_state_dict(hf_state)
    missing, unexpected = our_tower.load_state_dict(our_state, strict=False)
    assert not unexpected, f"Unexpected keys after remap: {sorted(unexpected)[:5]}"
    blocking_missing = [m for m in missing if not m.endswith(".inv_freq")]
    assert not blocking_missing, f"Missing keys: {sorted(blocking_missing)[:5]}"

    # Apply the new vision quant recipe (PE table → int8).
    quantize_vision_position_table(our_tower.vision_tower, verbose=True)

    with torch.no_grad():
        our_emb, our_mask = our_tower(pixel_values, pixel_position_ids)

    # ---- Compare ----
    assert torch.equal(hf_mask.bool(), our_mask.bool()), "pooler mask mismatch"
    cos = torch.nn.functional.cosine_similarity(
        hf_emb.flatten().float(), our_emb.flatten().float(), dim=0
    ).item()
    print(f"\nVision quant cosine sim (HF bf16 vs our PE-int8): {cos:.7f}")
    diff = (hf_emb - our_emb).abs()
    print(f"  max abs diff: {diff.max().item():.3e}")
    print(f"  mean abs diff: {diff.mean().item():.3e}")
    assert cos > 0.999, f"cosine_sim {cos} below 0.999 threshold for vision quant"


# ---------------------------------------------------------------------------
# Test 2 — keys saved with vision detached match the text-only snapshot.
# ---------------------------------------------------------------------------


def _tiny_recipe(hidden_size: int):
    """Tiny analogue of GEMMA4_31B_DEFAULT_RECIPE for unit-test models.

    Production recipe uses group_size=hidden_size (5376) for the per-axis
    embedding INT8 quant; that doesn't fit a 64-d test model. Functional
    shape is identical: INT8 per-axis embed, skip norms, INT4 elsewhere.
    """
    from executorch.examples.models.gemma4_31b.quant import (
        QuantConfig,
        QuantRecipe,
        QuantRule,
    )

    int4 = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
    int8_per_axis = QuantConfig(
        bits=8, group_size=hidden_size, symmetric=True, method="min_max"
    )
    return QuantRecipe(
        rules=[
            QuantRule(r"embed_tokens\.weight", int8_per_axis),
            QuantRule(r".*norm\.weight", None),
            QuantRule(r".*\.weight", int4),
        ]
    )


def _txt_keys_after_quantize(model: Gemma4_31B) -> tuple[set, set]:
    """quantize + flatten + read keys, mimicking what quantize_and_save.py writes
    for the text decoder when no vision tensors are added."""
    from executorch.examples.models.gemma4_31b.quant import quantize_model

    state_dict = quantize_model(model, _tiny_recipe(model.config.hidden_size))
    tensors_data, metadata = flatten_tensor_state_dict(state_dict)
    return set(tensors_data.keys()), set(json.loads(metadata.get("tensor_names", "[]")))


# Removed: `test_text_only_save_keys_match_snapshot` and
# `test_load_prequantized_text_only_no_vision_attached` \u2014 these tested the
# old text-only branch (build a model without vision_config). Vision is now
# mandatory; missing vision = error, so those scenarios are no longer valid.


def test_no_vision_silently_skips_when_vision_attached():
    """Build a model WITH vision attached (vision_config != None), then run
    the quantize_and_save 'detach + skip' path: the resulting saved-keys set
    must NOT contain any vision key.

    This guards the always-on vision behavior in quantize_and_save.py: the
    `--no-vision` CLI flag was removed in the design pivot, but the underlying
    detach pattern (which prevents the text-decoder recipe from quantizing
    vision linears) is still exercised here.
    """
    from executorch.examples.models.gemma4_31b.quant import quantize_model

    cfg = Gemma4_31BConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        num_global_key_value_heads=1,
        global_head_dim=64,
        attention_k_eq_v=True,
        sliding_window=8,
        max_seq_len=32,
        vision_config=Gemma4VisionConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=32,
            patch_size=4,
            pooling_kernel_size=2,
            position_embedding_size=32,
            standardize=True,
        ),
    )
    torch.manual_seed(0)
    model = Gemma4_31B(cfg).to(dtype=torch.bfloat16)
    for p in model.parameters():
        if p.device.type != "meta":
            p.data.normal_(0, 0.02)
    model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())

    # Sanity: vision is attached.
    assert hasattr(model, "vision_tower")
    assert hasattr(model, "embed_vision")

    # Mirror quantize_and_save.py's detach-then-quantize text-decoder flow:
    del model.vision_tower
    del model.embed_vision

    state_dict = quantize_model(model, _tiny_recipe(cfg.hidden_size))
    _, metadata = flatten_tensor_state_dict(state_dict)
    logical = set(json.loads(metadata["tensor_names"]))
    leaked = [
        k
        for k in logical
        if k.startswith("vision_tower.") or k.startswith("embed_vision.")
    ]
    assert not leaked, f"--no-vision still saved vision keys: {leaked[:5]}"


# ---------------------------------------------------------------------------
# Test 3 — has_vision_keys() detects both kinds of saves.
# ---------------------------------------------------------------------------


def test_has_vision_keys_text_only():
    """A safetensors with no vision keys → has_vision_keys returns False."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.safetensors")
        plain = {
            "layers.0.input_layernorm.weight": torch.randn(8, dtype=torch.bfloat16)
        }
        td, md = flatten_tensor_state_dict(plain)
        save_file(td, path, metadata=md)
        assert has_vision_keys(path) is False


def test_has_vision_keys_with_vision():
    """A safetensors with vision_tower.* → has_vision_keys returns True."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.safetensors")
        plain = {
            "embed_tokens.weight": torch.randn(16, 8, dtype=torch.bfloat16),
            "vision_tower.encoder.layers.0.input_layernorm.weight": torch.randn(
                4, dtype=torch.bfloat16
            ),
        }
        td, md = flatten_tensor_state_dict(plain)
        save_file(td, path, metadata=md)
        assert has_vision_keys(path) is True


# Removed: `_save_text_only_tiny_checkpoint` and
# `test_load_prequantized_text_only_no_vision_attached` \u2014 the text-only
# checkpoint case is no longer valid (vision is mandatory).


# ---------------------------------------------------------------------------
# Test 4 \u2014 install_int8_pe_dispatch + collect_vision_state_dict roundtrip.
# ---------------------------------------------------------------------------


def test_pe_int8_quantize_and_install_roundtrip():
    """quantize_vision_position_table + collect_vision_state_dict produce a
    state_dict that round-trips via install_int8_pe_dispatch + load."""
    cfg = Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        patch_size=4,
        pooling_kernel_size=2,
        position_embedding_size=16,
        standardize=True,
    )
    torch.manual_seed(0)
    tower = Gemma4_31BVisionTower(cfg, text_hidden_size=64).to(dtype=torch.bfloat16)
    for p in tower.parameters():
        p.data.normal_(0, 0.02)

    # Snapshot reference output PRE-quant.
    g = torch.Generator().manual_seed(0)
    pv = torch.rand(1, 16, cfg.patch_dim, generator=g, dtype=torch.bfloat16)
    coords = torch.arange(4)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    pp = torch.stack([xx.flatten(), yy.flatten()], -1).unsqueeze(0).long()
    with torch.no_grad():
        ref_emb, ref_mask = tower(pv, pp)

    # Quantize PE table in-place.
    quantize_vision_position_table(tower.vision_tower, verbose=False)
    assert hasattr(tower.vision_tower.patch_embedder, "_pet_int8")
    assert hasattr(tower.vision_tower.patch_embedder, "_pet_scale")
    assert not hasattr(tower.vision_tower.patch_embedder, "position_embedding_table")

    with torch.no_grad():
        post_emb, post_mask = tower(pv, pp)
    assert torch.equal(ref_mask, post_mask)
    cos = torch.nn.functional.cosine_similarity(
        ref_emb.flatten().float(), post_emb.flatten().float(), dim=0
    ).item()
    assert cos > 0.999, f"PE-int8 round-trip cosine {cos} too low"

    # Collect and reinstall on a fresh tower.
    state = collect_vision_state_dict(tower.vision_tower, tower.embed_vision)
    assert "vision_tower.patch_embedder._pet_int8" in state
    assert state["vision_tower.patch_embedder._pet_int8"].dtype == torch.int8
    assert "vision_tower.patch_embedder._pet_scale" in state
    assert state["vision_tower.patch_embedder._pet_scale"].dtype == torch.float32

    fresh = Gemma4_31BVisionTower(cfg, text_hidden_size=64).to(dtype=torch.bfloat16)
    install_int8_pe_dispatch(fresh.vision_tower, verbose=False)
    # Replace meta-buffers with real loaded ones.
    pe = fresh.vision_tower.patch_embedder
    pe._pet_int8 = state["vision_tower.patch_embedder._pet_int8"].clone()
    pe._pet_scale = state["vision_tower.patch_embedder._pet_scale"].clone()
    # Load the rest via load_state_dict (skip _pet_* — we already set them).
    # Build with our nested keys ("vision_tower.X" / "embed_vision.X")
    nested = {
        k: v
        for k, v in state.items()
        if not k.endswith("._pet_int8") and not k.endswith("._pet_scale")
    }
    missing, unexpected = fresh.load_state_dict(nested, strict=False)
    blocking_missing = [m for m in missing if "patch_embedder._pet_" not in m]
    # std_bias / std_scale present, encoder.rotary_emb.inv_freq is non-persistent → may be in `missing`
    blocking_missing = [m for m in blocking_missing if not m.endswith(".inv_freq")]
    assert not blocking_missing, f"Reinstall missing keys: {blocking_missing}"

    with torch.no_grad():
        reloaded_emb, reloaded_mask = fresh(pv, pp)
    assert torch.equal(reloaded_mask, post_mask)
    cos2 = torch.nn.functional.cosine_similarity(
        reloaded_emb.flatten().float(), post_emb.flatten().float(), dim=0
    ).item()
    assert cos2 > 0.99999, f"Reinstall round-trip cosine {cos2} too low"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
