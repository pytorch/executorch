# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test quantize-save-load roundtrip for quantized Qwen 3.5 MoE.

Creates a tiny model (no downloads needed), quantizes it, saves to safetensors,
loads back via the production load_prequantized_model codepath, and verifies
the loaded model produces identical forward outputs.

Requires CUDA (Int4TilePackedTo4dTensor needs _convert_weight_to_int4pack).

Usage:
    python -m pytest examples/models/qwen3_5_moe/test_quantize_roundtrip.py -v
"""

import json
import os
import tempfile
import unittest

import torch

from executorch.examples.models.qwen3_5_moe.export import (
    _materialize_buffers,
    _quantize,
    load_prequantized_model,
)
from executorch.examples.models.qwen3_5_moe.model import Qwen35MoE, Qwen35MoEConfig
from executorch.examples.models.qwen3_5_moe.quantize_and_save import (
    save_quantized_model,
)


TINY_CONFIG = Qwen35MoEConfig(
    vocab_size=256,
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_kv_heads=2,
    head_dim=64,
    partial_rotary_factor=0.25,
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    linear_key_head_dim=64,
    linear_value_head_dim=64,
    linear_conv_kernel_dim=4,
    num_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=128,
    shared_expert_intermediate_size=128,
    full_attention_interval=2,
    rms_norm_eps=1e-6,
    rope_theta=10_000.0,
    max_seq_len=64,
)


def _make_quantized_model(qlinear, qembedding, group_size, hqq=False):
    """Create a tiny model with random weights and quantize it."""
    torch.manual_seed(42)
    model = Qwen35MoE(TINY_CONFIG)
    model.to(dtype=torch.bfloat16)
    for p in model.parameters():
        if p.device.type != "meta":
            p.data.normal_(0, 0.02)
    model.eval()

    class Args:
        pass

    args = Args()
    args.qlinear = qlinear
    args.qembedding = qembedding
    args.qlinear_group_size = group_size
    args.qlinear_packing_format = "tile_packed_to_4d"

    args.hqq = hqq
    _quantize(model, TINY_CONFIG, args)

    return model


def _save_bundle(model, output_dir):
    """Save a quantized model as a bundle (model.safetensors + config.json).

    Uses the production save_quantized_model for weights, and writes a
    config.json from TINY_CONFIG so load_prequantized_model can read it.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_quantized_model(model, os.path.join(output_dir, "model.safetensors"))

    # Write config.json matching TINY_CONFIG fields that from_hf_config reads
    config_dict = {
        "vocab_size": TINY_CONFIG.vocab_size,
        "hidden_size": TINY_CONFIG.hidden_size,
        "num_hidden_layers": TINY_CONFIG.num_hidden_layers,
        "num_attention_heads": TINY_CONFIG.num_attention_heads,
        "num_key_value_heads": TINY_CONFIG.num_kv_heads,
        "head_dim": TINY_CONFIG.head_dim,
        "partial_rotary_factor": TINY_CONFIG.partial_rotary_factor,
        "linear_num_key_heads": TINY_CONFIG.linear_num_key_heads,
        "linear_num_value_heads": TINY_CONFIG.linear_num_value_heads,
        "linear_key_head_dim": TINY_CONFIG.linear_key_head_dim,
        "linear_value_head_dim": TINY_CONFIG.linear_value_head_dim,
        "linear_conv_kernel_dim": TINY_CONFIG.linear_conv_kernel_dim,
        "num_experts": TINY_CONFIG.num_experts,
        "num_experts_per_tok": TINY_CONFIG.num_experts_per_tok,
        "moe_intermediate_size": TINY_CONFIG.moe_intermediate_size,
        "shared_expert_intermediate_size": TINY_CONFIG.shared_expert_intermediate_size,
        "full_attention_interval": TINY_CONFIG.full_attention_interval,
        "rms_norm_eps": TINY_CONFIG.rms_norm_eps,
        "rope_theta": TINY_CONFIG.rope_theta,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f)


class TestQuantizeRoundtrip(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

    def _test_roundtrip(self, qlinear, qembedding, group_size, hqq=False):
        """Test: quantize -> save -> load_prequantized_model -> forward
        produces same output as quantize -> forward.
        """
        import executorch.backends.cuda.triton.kernels  # noqa: F401

        # Quantize and save bundle (before moving to CUDA, matching production flow)
        model_a = _make_quantized_model(qlinear, qembedding, group_size, hqq)
        with tempfile.TemporaryDirectory() as tmpdir:
            _save_bundle(model_a, tmpdir)

            # Path A: materialize, move to CUDA, forward
            _materialize_buffers(model_a, TINY_CONFIG)
            model_a.to(device="cuda")

            torch.manual_seed(99)
            tokens = torch.randint(0, TINY_CONFIG.vocab_size, (1, 4), device="cuda")
            input_pos = torch.arange(4, device="cuda")

            with torch.no_grad():
                output_a = model_a(tokens, input_pos)
            del model_a

            # Path B: load via production codepath, materialize, forward
            model_b, _ = load_prequantized_model(
                tmpdir, max_seq_len=TINY_CONFIG.max_seq_len
            )

        _materialize_buffers(model_b, TINY_CONFIG)
        model_b.to(device="cuda")

        with torch.no_grad():
            output_b = model_b(tokens, input_pos)

        self.assertTrue(
            torch.equal(output_a, output_b),
            f"Outputs differ: max diff = {(output_a - output_b).abs().max().item()}",
        )

    def test_4w_8w_gs32(self):
        """Roundtrip: 4w linear + 8w embedding, group_size=32."""
        self._test_roundtrip("4w", "8w", 32)

    def test_4w_gs32(self):
        """Roundtrip: 4w linear only, group_size=32."""
        self._test_roundtrip("4w", None, 32)

    def test_4w_8w_gs128(self):
        """Roundtrip: 4w linear + 8w embedding, group_size=128."""
        self._test_roundtrip("4w", "8w", 128)

    def test_4w_8w_gs128_hqq(self):
        """Roundtrip: 4w linear + 8w embedding, group_size=128, HQQ experts."""
        self._test_roundtrip("4w", "8w", 128, hqq=True)

    def test_load_rejects_corrupted_checkpoint(self):
        """load_prequantized_model raises on corrupted/mismatched checkpoint.

        Saves a valid checkpoint, then replaces it with one that has a
        renamed key (simulating a version mismatch). The loader should
        raise RuntimeError instead of silently producing a broken model.
        """
        from safetensors import safe_open
        from safetensors.torch import save_file

        model = _make_quantized_model("4w", None, 32)

        with tempfile.TemporaryDirectory() as tmpdir:
            _save_bundle(model, tmpdir)
            path = os.path.join(tmpdir, "model.safetensors")

            # Load, rename a key to simulate version mismatch, re-save
            with safe_open(path, framework="pt", device="cpu") as f:
                header = f.metadata()
                tensors = {k: f.get_tensor(k) for k in f.keys()}

            # Rename norm.weight -> norm.BOGUS (missing + unexpected)
            tensors["norm.BOGUS"] = tensors.pop("norm.weight")
            save_file(tensors, path, metadata=header)

            with self.assertRaises(RuntimeError):
                load_prequantized_model(tmpdir, max_seq_len=TINY_CONFIG.max_seq_len)


if __name__ == "__main__":
    unittest.main()
