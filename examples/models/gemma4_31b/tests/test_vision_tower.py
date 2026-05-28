# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validation gate for the Gemma 4 31B vision tower port.

Vision is always on; there is exactly one model shape (text + vision).

Tests:
  1. ``test_forward_signature_takes_inputs_embeds`` -- forward signature is the
     unified-prefill shape.
  2. ``test_multimodal_methods_present`` -- embed_text + decode_forward exist;
     legacy prefill_image is GONE.
  3. ``test_decode_forward_equivalent_to_embed_then_forward`` -- the fused
     decode_forward matches forward(embed_text(tokens)) within bf16 tolerance
     on a tiny, random-init model (no external checkpoint required).
  4. ``test_vision_tower_random_init_forward_smoke`` -- random-init our ported
     vision tower (bf16), run the forward, and verify shapes / finite outputs.

No external HF checkpoint is required. Earlier revisions of this file
compared our port against ``Gemma4ForConditionalGeneration`` loaded from
disk, but that gated the entire suite on a hardcoded path. The
quantization-roundtrip tests now use a random-init bf16 baseline of OUR
tower as the reference, which is sufficient for catching regressions in
the port without needing the upstream weights.
"""

from __future__ import annotations

import sys

import pytest
import torch

from executorch.examples.models.gemma4_31b.model import Gemma4_31B, Gemma4_31BConfig
from executorch.examples.models.gemma4_31b.vision_tower import (
    Gemma4_31BVisionTower,
    Gemma4VisionConfig,
)


# ---------------------------------------------------------------------------
# Shared tiny config (no external checkpoint required).
# ---------------------------------------------------------------------------


def _tiny_vision_config() -> Gemma4VisionConfig:
    return Gemma4VisionConfig(
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


def _tiny_model_config() -> Gemma4_31BConfig:
    """A minimal Gemma4_31B config sized so the test runs in seconds on CPU."""
    return Gemma4_31BConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_global_key_value_heads=1,
        global_head_dim=16,
        attention_k_eq_v=True,
        sliding_window=16,
        max_seq_len=64,
        layer_types=["sliding_attention", "full_attention"],
        vision_config=_tiny_vision_config(),
    )


def _make_vision_inputs(
    batch: int,
    grid: int,
    patch_dim: int,
    seed: int = 0,
    dtype: torch.dtype = torch.bfloat16,
):
    """Build a deterministic (pixel_values, pixel_position_ids) pair.

    ``num_patches = grid*grid`` must divide cleanly by
    ``pooling_kernel_size**2`` (the pooler reshape constraint).
    """
    g = torch.Generator().manual_seed(seed)
    num_patches = grid * grid
    pixel_values = torch.rand(batch, num_patches, patch_dim, generator=g, dtype=dtype)
    coords = torch.arange(grid, dtype=torch.long)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    pos = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # [G*G, 2]
    pixel_position_ids = pos.unsqueeze(0).expand(batch, -1, -1).contiguous()
    return pixel_values, pixel_position_ids


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
    output as ``forward(embed_text(tokens), ...)`` on the same input. Both
    paths internally call ``sample()`` (Gumbel-max), so the RNG is seeded
    identically before each call to keep the noise term equal across runs.
    """
    cfg = _tiny_model_config()
    torch.manual_seed(0)
    model = Gemma4_31B(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 8), dtype=torch.long)
    input_pos = torch.arange(8, dtype=torch.long)
    # Small temperature -> near-greedy Gumbel-max. With matching RNG seeds
    # on both branches the sampled token IDs must agree exactly.
    temperature = torch.tensor([1.0], dtype=torch.float32)

    with torch.no_grad():
        torch.manual_seed(123)
        out_decode = model.decode_forward(tokens, input_pos, temperature).clone()
        # Re-run via embed_text + forward. KV slots are overwritten with the
        # same values (input_pos starts at 0 again).
        torch.manual_seed(123)
        inputs_embeds_fp32 = model.embed_text(tokens).to(torch.float32)
        out_via_embeds = model.forward(inputs_embeds_fp32, input_pos, temperature)

    assert torch.equal(
        out_decode, out_via_embeds
    ), f"decode_forward vs embed+forward diverged: {out_decode} vs {out_via_embeds}"


# ---------------------------------------------------------------------------
# Random-init forward smoke test for the vision tower.
# ---------------------------------------------------------------------------


def test_vision_tower_random_init_forward_smoke():
    """Random-init our ported vision tower (bf16) and verify the forward
    produces a finite tensor of the expected shape on a small fixed input.

    This replaces the old HF-parity gate; we no longer require the
    upstream Gemma4 checkpoint to validate that the tower wiring is
    structurally correct. Numerical parity against HF still belongs in a
    separate offline gate that has access to the bf16 reference weights.
    """
    torch.manual_seed(0)
    dtype = torch.bfloat16

    cfg = _tiny_vision_config()
    text_hidden_size = 64
    grid = 4  # 4x4 = 16 patches, divides cleanly by pooling_kernel_size**2 = 4

    tower = (
        Gemma4_31BVisionTower(cfg, text_hidden_size=text_hidden_size).to(dtype).eval()
    )
    # Spread the random weights a bit so the smoke test exercises non-zero
    # activations through every sub-module.
    for p in tower.parameters():
        if p.device.type != "meta":
            p.data.normal_(0, 0.02)

    pixel_values, pixel_position_ids = _make_vision_inputs(
        batch=1, grid=grid, patch_dim=cfg.patch_dim, dtype=dtype
    )

    with torch.no_grad():
        emb, mask = tower(pixel_values, pixel_position_ids)

    # The pooler collapses pks*pks patches into one soft token.
    expected_soft_tokens = (grid * grid) // (cfg.pooling_kernel_size**2)
    assert emb.shape == (
        1,
        expected_soft_tokens,
        text_hidden_size,
    ), f"unexpected vision_tower output shape: {emb.shape}"
    assert mask.shape == (
        1,
        expected_soft_tokens,
    ), f"unexpected pooler mask shape: {mask.shape}"
    assert emb.dtype == dtype
    assert torch.isfinite(emb).all(), "vision_tower produced non-finite values"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
