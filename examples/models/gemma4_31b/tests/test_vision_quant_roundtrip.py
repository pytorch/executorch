# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validation gates for the gemma4_31b vision-tower quantization recipe.

All tests build a random-init bf16 baseline of our own model/tower as
the reference. No external HF checkpoint is required.

Tests:

* ``test_pe_int8_quantize_and_install_roundtrip`` -- snapshot a random-
  init bf16 tower's output, apply ``quantize_vision_position_table``,
  collect the state dict, reinstall it on a freshly-built tower via
  ``install_int8_pe_dispatch``, and verify the output round-trips with
  cosine_sim > 0.999.

* ``test_unified_recipe_preserves_vision_bf16_and_quantizes_pe`` --
  build a tiny Gemma4_31B with vision attached, run ``quantize_model``
  with the unified recipe, and verify the vision linears stay bf16
  while the PE table is swapped to int8 buffers.

* ``test_has_vision_keys_*`` -- ``has_vision_keys`` sniff test on plain
  safetensors files.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest
import torch
import torch.nn as nn

from executorch.examples.models.gemma4_31b.model import Gemma4_31B, Gemma4_31BConfig
from executorch.examples.models.gemma4_31b.pack_vision import (
    collect_vision_state_dict,
    has_vision_keys,
    install_int8_pe_dispatch,
    quantize_vision_position_table,
)
from executorch.examples.models.gemma4_31b.vision_tower import (
    Gemma4_31BVisionTower,
    Gemma4VisionConfig,
)
from safetensors.torch import save_file
from torchao.prototype.safetensors.safetensors_support import flatten_tensor_state_dict


# ---------------------------------------------------------------------------
# Test 1 -- unified recipe leaves vision linears bf16 and quantizes the PE table.
# ---------------------------------------------------------------------------


def _tiny_recipe(hidden_size: int):
    """Tiny analogue of GEMMA4_31B_DEFAULT_RECIPE for unit-test models.

    Production recipe uses group_size=hidden_size (5376) for the per-axis
    embedding INT8 quant; that doesn't fit a 64-d test model. Functional
    shape is identical: INT8 per-axis embed, skip vision side + norms,
    INT4 elsewhere.
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
            # Vision modality stays bf16; PE table is quantized explicitly before
            # calling quantize_model.
            QuantRule(r"vision_tower\..*", None),
            QuantRule(r"embed_vision\..*", None),
            QuantRule(r".*norm\.weight", None),
            QuantRule(r".*\.weight", int4),
        ],
    )


def test_unified_recipe_preserves_vision_bf16_and_quantizes_pe():
    """Build a model WITH vision attached, run the unified recipe, and check:

    * vision_tower.* and embed_vision.* linear weights are saved (no
      detach hack required);
    * NO vision linear is quantized to Int4Tensor /
      IntxUnpackedToInt8Tensor (they all stay bf16);
    * the PE table has been swapped to int8 buffers (_pet_int8 +
      _pet_scale) by quantize_model itself.

    The unified ``quantize_model`` API handles vision + text in a single
    pass, replacing the old ``del model.vision_tower`` pattern.
    """
    from executorch.examples.models.gemma4_31b.quant import quantize_model
    from torchao.quantization import IntxUnpackedToInt8Tensor
    from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor

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

    quantize_vision_position_table(model.vision_tower)
    state_dict = quantize_model(model, _tiny_recipe(cfg.hidden_size))

    # Vision-side linears must be present AND must stay bf16 (plain Tensor,
    # not a quantized subclass).
    vision_param_keys = [
        k
        for k in state_dict
        if (k.startswith("vision_tower.") or k.startswith("embed_vision."))
        and k.endswith(".weight")
    ]
    assert vision_param_keys, "expected vision_tower / embed_vision weights to be saved"
    for k in vision_param_keys:
        v = state_dict[k]
        assert not isinstance(v, (Int4Tensor, IntxUnpackedToInt8Tensor)), (
            f"vision weight {k} was quantized ({type(v).__name__}); "
            "the vision_tower / embed_vision recipe rules should keep it bf16"
        )
        assert (
            v.dtype == torch.bfloat16
        ), f"vision weight {k} dtype is {v.dtype}, expected bfloat16"

    # PE table is swapped to int8 buffers by quantize_model.
    assert "vision_tower.patch_embedder._pet_int8" in state_dict
    assert state_dict["vision_tower.patch_embedder._pet_int8"].dtype == torch.int8
    assert "vision_tower.patch_embedder._pet_scale" in state_dict
    assert state_dict["vision_tower.patch_embedder._pet_scale"].dtype == torch.float32
    # Sanity: the bf16 PE Parameter is gone from the saved keys.
    assert "vision_tower.patch_embedder.position_embedding_table" not in state_dict


# ---------------------------------------------------------------------------
# Test 2 -- has_vision_keys() detects both kinds of saves.
# ---------------------------------------------------------------------------


def test_has_vision_keys_text_only():
    """A safetensors with no vision keys -> has_vision_keys returns False."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.safetensors")
        plain = {
            "layers.0.input_layernorm.weight": torch.randn(8, dtype=torch.bfloat16)
        }
        td, md = flatten_tensor_state_dict(plain)
        save_file(td, path, metadata=md)
        assert has_vision_keys(path) is False


def test_has_vision_keys_with_vision():
    """A safetensors with vision_tower.* -> has_vision_keys returns True."""
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


# ---------------------------------------------------------------------------
# Test 3 -- quantize PE table + reinstall round-trips against the bf16 ref.
# ---------------------------------------------------------------------------


def test_pe_int8_quantize_and_install_roundtrip():
    """End-to-end PE-int8 round-trip on a random-init bf16 tower.

    Snapshot the bf16 reference output, apply
    ``quantize_vision_position_table`` in place, then collect the state
    dict and reinstall it on a freshly-built tower via
    ``install_int8_pe_dispatch`` + ``load_state_dict``. Cosine sim vs the
    bf16 reference must exceed 0.999 after the int8 swap, and the
    reload-then-forward path must match the in-place quantized forward to
    > 0.99999.
    """
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
    # Load the rest via load_state_dict (skip _pet_* -- we already set them).
    nested = {
        k: v
        for k, v in state.items()
        if not k.endswith("._pet_int8") and not k.endswith("._pet_scale")
    }
    missing, unexpected = fresh.load_state_dict(nested, strict=False)
    blocking_missing = [m for m in missing if "patch_embedder._pet_" not in m]
    # encoder.rotary_emb.inv_freq is non-persistent -> may be in `missing`
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
