# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validation gate for the Gemma 4 31B vision tower port.

Updated for the 4-method unified contract (orchestrator pin #4):
  forward(inputs_embeds, input_pos, temperature)  -> exported as `prefill`
  decode_forward(tokens, input_pos, temperature)  -> exported as `decode`
  embed_text(tokens) -> bf16                      -> exported as `embed_text`
  Gemma4_31BVisionTower                           -> exported as `vision_encoder`

Tests:
  1. ``test_state_dict_keys_with_vision_is_strict_superset`` — vision attach
     adds exactly +356 keys (no text-key drift).
  2. ``test_forward_signature_takes_inputs_embeds`` — forward signature is the
     new unified-prefill shape.
  3. ``test_multimodal_methods_present`` — embed_text + decode_forward exist;
     legacy prefill_image is GONE.
  4. ``test_decode_forward_equivalent_to_embed_then_forward`` — the fused
     decode_forward matches forward(embed_text(tokens)) within bf16 tolerance.
  5. ``test_vision_tower_matches_hf`` — cosine_sim > 0.99999 vs HF baseline.

The HF reference is built by loading ONLY the vision sub-modules of
``Gemma4ForConditionalGeneration`` (so we don't pay the LM-loading cost) and
wrapping them in the same export-safe shape as
``examples/models/gemma4/export_gemma4.py::_HFVisionEncoderWithProjection``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch

# Make the repo importable as `executorch.*` even when conda picks up an older
# site-packages copy. The vision_tower module imports through this same prefix.
_REPO_ROOT = "/home/gasoonjia/executorch"
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from executorch.examples.models.gemma4_31b.model import (  # noqa: E402
    Gemma4_31B,
    Gemma4_31BConfig,
)
from executorch.examples.models.gemma4_31b.vision_tower import (  # noqa: E402
    Gemma4_31BVisionTower,
    Gemma4VisionConfig,
    hf_vision_key_map,
    hf_vision_per_layer_key_map,
)


HF_MODEL_DIR = "/home/gasoonjia/models/gemma-4-31B"
SNAPSHOT_PATH = Path(__file__).parent / "_text_only_state_dict_keys.json"


# ---------------------------------------------------------------------------
# Test 1: vision attach is additive (no text-key drift) — ADD-ONLY guarantee.
# ---------------------------------------------------------------------------


def _expected_text_only_keys() -> set[str]:
    with open(SNAPSHOT_PATH, "r") as f:
        return set(json.load(f))


def test_state_dict_keys_with_vision_is_strict_superset():
    """vision_config != None → adds exactly the vision_tower / embed_vision keys.

    Even after the design pivot, this still holds: the model.py changes
    affect forward semantics, not the on-checkpoint key layout. State dict =
    833 text keys (snapshot) + 356 vision keys (attached when vision_config
    is parsed from the HF config.json).
    """
    cfg = Gemma4_31BConfig.from_hf_config(os.path.join(HF_MODEL_DIR, "config.json"))
    assert cfg.vision_config is not None
    cfg.max_seq_len = 4096
    with torch.device("meta"):
        model = Gemma4_31B(cfg)
    keys = set(model.state_dict().keys())
    expected_text = _expected_text_only_keys()
    extras = keys - expected_text
    missing = expected_text - keys
    assert not missing, f"Vision build dropped text keys: {sorted(missing)[:10]}"
    bad_extras = [
        k
        for k in extras
        if not (k.startswith("vision_tower.") or k.startswith("embed_vision."))
    ]
    assert not bad_extras, f"Unexpected non-vision extras: {bad_extras[:10]}"
    assert len(extras) == 356, f"Expected 356 vision keys, got {len(extras)}"


# ---------------------------------------------------------------------------
# Method shape contract: 4-method unified design (orchestrator pin #4).
# ---------------------------------------------------------------------------


def test_forward_signature_takes_inputs_embeds():
    """`Gemma4_31B.forward` is the unified prefill: takes inputs_embeds, NOT tokens."""
    import inspect

    sig = inspect.signature(Gemma4_31B.forward)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == [
        "self",
        "inputs_embeds",
        "input_pos",
        "temperature",
    ], f"forward signature: {[p.name for p in params]}"
    assert params[3].default is None, "temperature must default to None"


def test_multimodal_methods_present():
    """Confirm the 4-method contract: forward + decode_forward + embed_text exist;
    the legacy prefill_image method is removed."""
    import inspect

    for name in ("embed_text", "decode_forward", "forward"):
        assert hasattr(Gemma4_31B, name), f"Gemma4_31B missing required method: {name}"
    assert not hasattr(
        Gemma4_31B, "prefill_image"
    ), "prefill_image should be removed in the 4-method contract"
    embed_sig = inspect.signature(Gemma4_31B.embed_text)
    assert [p.name for p in embed_sig.parameters.values()] == [
        "self",
        "tokens",
    ], f"embed_text signature: {embed_sig}"
    decode_sig = inspect.signature(Gemma4_31B.decode_forward)
    names = [p.name for p in decode_sig.parameters.values()]
    assert names == [
        "self",
        "tokens",
        "input_pos",
        "temperature",
    ], f"decode_forward signature: {decode_sig}"


def test_decode_forward_equivalent_to_embed_then_forward():
    """Eager equivalence: ``decode_forward(tokens, ...)`` must produce the same
    output as ``forward(embed_text(tokens), ...)`` (within bf16 tolerance).
    Validates that the runner can pipe text-only flow either way and get the
    same answer.
    """
    cfg = Gemma4_31BConfig.from_hf_config(os.path.join(HF_MODEL_DIR, "config.json"))
    # Vision is mandatory now \u2014 swap in a tiny vision_config so the model
    # construction stays cheap without requiring HF's full 1152-dim tower.
    cfg.vision_config = Gemma4VisionConfig(
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
    cfg.max_seq_len = 64
    cfg.num_hidden_layers = 2
    cfg.layer_types = ["sliding_attention", "full_attention"]
    cfg.vocab_size = 256
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.head_dim = 16
    cfg.num_global_key_value_heads = 1
    cfg.global_head_dim = 16
    torch.manual_seed(0)
    model = Gemma4_31B(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 8), dtype=torch.long)
    input_pos = torch.arange(8, dtype=torch.long)

    with torch.no_grad():
        out_decode = model.decode_forward(tokens, input_pos, temperature=None).clone()
        # Re-run via embed_text + forward. KV slots are overwritten with the
        # same values (input_pos starts at 0 again).
        inputs_embeds_fp32 = model.embed_text(tokens).to(torch.float32)
        out_via_embeds = model.forward(inputs_embeds_fp32, input_pos, temperature=None)

    diff = (out_decode - out_via_embeds).abs().max().item()
    # Tolerance covers bf16 round-trip in embed_text (≈ 3 decimal digits of
    # precision × 2 transformer layers ≈ 5e-3 worst-case logit noise).
    assert torch.allclose(
        out_decode, out_via_embeds, atol=1e-2, rtol=1e-2
    ), f"decode_forward vs embed+forward diverged: max diff {diff}"


# ---------------------------------------------------------------------------
# Test 3: HF-baseline cosine-sim parity for the vision tower forward.
# ---------------------------------------------------------------------------


def _hf_to_our_vision_state_dict(hf_state: dict) -> dict:
    """Remap HF vision_tower / embed_vision keys to ours.

    Used by the test wrapper to instantiate our ported tower with HF's exact
    weights so the cosine-sim comparison is meaningful.
    """
    fixed = hf_vision_key_map()
    per_layer = hf_vision_per_layer_key_map()
    out = {}
    import re as _re

    for k, v in hf_state.items():
        # Normalize: HF model.model.* prefix or top-level.
        norm = k
        if not (
            norm.startswith("model.vision_tower.")
            or norm.startswith("model.embed_vision.")
        ):
            norm = "model." + norm if not norm.startswith("model.") else norm
        if norm in fixed:
            out[fixed[norm]] = v
            continue
        matched = False
        for hf_pat, model_pat in per_layer.items():
            regex = _re.escape(hf_pat).replace(r"\{\}", r"(\d+)")
            m = _re.fullmatch(regex, norm)
            if m:
                out[model_pat.replace("{}", m.group(1), 1)] = v
                matched = True
                break
        if not matched:
            # silently skip — keeps the helper robust to extra HF buffers.
            pass
    return out


def _build_hf_vision_wrapper(dtype: torch.dtype):
    """Load just the HF vision tower + embed_vision (NOT the LM)."""
    from transformers import Gemma4ForConditionalGeneration

    # Loading the full model is expensive but unavoidable on this checkpoint
    # because there is no public from_pretrained() for the vision sub-tower
    # alone. We immediately drop the language model to free RAM.
    model = Gemma4ForConditionalGeneration.from_pretrained(
        HF_MODEL_DIR,
        dtype=dtype,
        device_map="cpu",
    )
    vision_tower = model.model.vision_tower
    embed_vision = model.model.embed_vision

    class _HFVisionWrapper(torch.nn.Module):
        """Mirror of ``examples/models/gemma4/export_gemma4.py::_HFVisionEncoderWithProjection``."""

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
            embeddings = self.embed_vision(hidden_states)
            return embeddings, pooler_mask

    wrapper = _HFVisionWrapper(vision_tower, embed_vision).eval()
    text_hidden = model.config.text_config.hidden_size

    # Snapshot HF state dict for our model BEFORE we drop `model`.
    hf_state = {}
    for k, v in vision_tower.state_dict().items():
        hf_state["model.vision_tower." + k] = v
    for k, v in embed_vision.state_dict().items():
        hf_state["model.embed_vision." + k] = v

    del model
    import gc as _gc

    _gc.collect()

    return wrapper, hf_state, text_hidden


def _make_inputs(
    batch: int,
    grid: int,
    patch_dim: int,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
):
    """Build a deterministic (pixel_values, pixel_position_ids) pair.

    For HF parity the only constraint is that ``num_patches`` divides cleanly by
    ``pooling_kernel_size**2`` (= 9). We set num_patches = grid*grid and arrange
    the position_ids on a grid so the pooler's reshape works.
    """
    g = torch.Generator().manual_seed(seed)
    num_patches = grid * grid
    pixel_values = torch.rand(batch, num_patches, patch_dim, generator=g, dtype=dtype)
    # 2D positions on a grid_xy x grid_xy lattice.
    coords = torch.arange(grid, dtype=torch.long)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    pos = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # [G*G, 2]
    pixel_position_ids = pos.unsqueeze(0).expand(batch, -1, -1).contiguous()
    return pixel_values, pixel_position_ids


def test_vision_tower_matches_hf():
    """Cosine sim > 0.99999 vs HF baseline on a small fixed input."""
    torch.manual_seed(0)
    dtype = torch.float32

    # Build inputs first (cheap) so a fast failure surfaces before HF load.
    # max_patches = 36 = 6x6 grid → output_length = 36 / 9 = 4 soft tokens.
    cfg = Gemma4VisionConfig.from_hf_config(os.path.join(HF_MODEL_DIR, "config.json"))
    pixel_values, pixel_position_ids = _make_inputs(
        batch=1, grid=6, patch_dim=cfg.patch_dim, seed=0, dtype=dtype
    )

    # ---- HF reference ----
    hf_wrapper, hf_state, text_hidden = _build_hf_vision_wrapper(dtype)
    with torch.no_grad():
        hf_emb, hf_mask = hf_wrapper(pixel_values, pixel_position_ids)

    # ---- Our ported tower ----
    our_tower = (
        Gemma4_31BVisionTower(cfg, text_hidden_size=text_hidden).to(dtype).eval()
    )
    our_state = _hf_to_our_vision_state_dict(hf_state)
    missing, unexpected = our_tower.load_state_dict(our_state, strict=False)
    assert not unexpected, f"Unexpected keys after remap: {sorted(unexpected)[:5]}"
    # std_bias / std_scale are buffers; they must be present.
    blocking_missing = [m for m in missing if not m.endswith(".inv_freq")]
    assert (
        not blocking_missing
    ), f"Missing keys after remap: {sorted(blocking_missing)[:5]}"

    with torch.no_grad():
        our_emb, our_mask = our_tower(pixel_values, pixel_position_ids)

    # ---- Compare ----
    assert (
        hf_emb.shape == our_emb.shape
    ), f"shape: hf={hf_emb.shape} ours={our_emb.shape}"
    assert (
        hf_mask.shape == our_mask.shape
    ), f"mask shape: hf={hf_mask.shape} ours={our_mask.shape}"
    assert torch.equal(
        hf_mask.bool(), our_mask.bool()
    ), f"pooler mask mismatch:\n hf={hf_mask}\n ours={our_mask}"

    diff = (hf_emb - our_emb).abs()
    print(f"\nMax abs diff: {diff.max().item():.3e}")
    print(f"Mean abs diff: {diff.mean().item():.3e}")
    cos = torch.nn.functional.cosine_similarity(
        hf_emb.flatten().float(), our_emb.flatten().float(), dim=0
    ).item()
    print(f"Cosine similarity: {cos:.7f}")
    assert cos > 0.99999, f"cosine_sim {cos} below 0.99999 threshold"
    assert torch.allclose(
        hf_emb, our_emb, atol=1e-4, rtol=1e-4
    ), f"allclose(atol=1e-4, rtol=1e-4) failed; max diff {diff.max().item():.3e}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
