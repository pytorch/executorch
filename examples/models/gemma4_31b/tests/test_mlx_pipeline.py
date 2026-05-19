# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end MLX backend tests for the Gemma 4 31B-IT pipeline.

Tests quantize → save → load → pack-for-MLX on a tiny model.
No CUDA or MLX hardware required.

Usage:
    python -m pytest examples/models/gemma4_31b/tests/test_mlx_pipeline.py -v
"""

import json
import os
import tempfile
import unittest

import torch
import torch.nn as nn

from executorch.examples.models.gemma4_31b.model import Gemma4_31B
from executorch.examples.models.gemma4_31b.quant import (
    DEFAULT_MLX_PACKERS,
    pack_model,
    QuantConfig,
    quantize_model,
    QuantRecipe,
    QuantRule,
)
from executorch.examples.models.gemma4_31b.tests.test_pipeline import (
    build_random_tiny_model,
    config_dict,
    save_checkpoint,
    TINY_CONFIG,
)

_INT4 = QuantConfig(bits=4, group_size=32, symmetric=True, method="min_max")
_INT8 = QuantConfig(bits=8, group_size=32, symmetric=True, method="min_max")
_INT8_PER_AXIS = QuantConfig(
    bits=8, group_size=TINY_CONFIG.hidden_size, symmetric=True, method="min_max"
)
_EDGE_LAYERS = set(range(3))

TINY_SENSITIVE_RECIPE = QuantRecipe(
    rules=[
        QuantRule(r"embed_tokens\.weight", _INT8_PER_AXIS),
        QuantRule(r".*norm\.weight", None),
        QuantRule(r".*\.(v_proj|down_proj)\.weight", _INT8, layers=_EDGE_LAYERS),
        QuantRule(r".*\.weight", _INT4),
    ]
)


class TestMlxPipeline(unittest.TestCase):
    """End-to-end: quantize → pack for MLX → forward."""

    def test_pack_for_mlx(self):
        """Quantize with sensitive recipe, pack for MLX, no meta weights."""
        model = build_random_tiny_model()
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
        state_dict = quantize_model(model, TINY_SENSITIVE_RECIPE)

        with torch.device("meta"):
            model = Gemma4_31B(TINY_CONFIG)
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
        pack_model(model, state_dict, DEFAULT_MLX_PACKERS)

        for fqn, p in model.named_parameters():
            self.assertNotEqual(p.device.type, "meta", f"Weight '{fqn}' still on meta")

    def test_forward_after_pack(self):
        """Model produces valid output after MLX packing."""
        model = build_random_tiny_model()
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
        state_dict = quantize_model(model, TINY_SENSITIVE_RECIPE)

        with torch.device("meta"):
            model = Gemma4_31B(TINY_CONFIG)
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
        pack_model(model, state_dict, DEFAULT_MLX_PACKERS)
        model.eval()

        from executorch.examples.models.gemma4_31b.model import (
            materialize_runtime_buffers,
        )

        materialize_runtime_buffers(model, dtype=torch.bfloat16)

        tokens = torch.randint(0, TINY_CONFIG.vocab_size, (1, 1))
        input_pos = torch.tensor([0], dtype=torch.long)
        temp = torch.tensor([1e-6], dtype=torch.float32)

        with torch.no_grad():
            out = model(tokens, input_pos, temp)

        self.assertEqual(out.shape, torch.Size([1, 1]))
        self.assertFalse(torch.isnan(out).any())

    def test_multi_token_forward(self):
        model = build_random_tiny_model()
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
        state_dict = quantize_model(model, TINY_SENSITIVE_RECIPE)

        with torch.device("meta"):
            model = Gemma4_31B(TINY_CONFIG)
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
        pack_model(model, state_dict, DEFAULT_MLX_PACKERS)
        model.eval()

        from executorch.examples.models.gemma4_31b.model import (
            materialize_runtime_buffers,
        )

        materialize_runtime_buffers(model, dtype=torch.bfloat16)

        seq_len = 4
        tokens = torch.randint(0, TINY_CONFIG.vocab_size, (1, seq_len))
        input_pos = torch.arange(seq_len, dtype=torch.long)
        temp = torch.tensor([1e-6], dtype=torch.float32)

        with torch.no_grad():
            out = model(tokens, input_pos, temp)

        self.assertEqual(out.shape, torch.Size([1, 1]))
        self.assertFalse(torch.isnan(out).any())

    def test_source_transforms_forward(self):
        """Model produces valid output after MLX source transforms."""
        model = build_random_tiny_model()
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
        state_dict = quantize_model(model, TINY_SENSITIVE_RECIPE)

        with torch.device("meta"):
            model = Gemma4_31B(TINY_CONFIG)
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
        pack_model(model, state_dict, DEFAULT_MLX_PACKERS)
        model.eval()

        from executorch.examples.models.gemma4_31b.mlx_source_transformations import (
            mlx_source_transformations,
        )
        from executorch.examples.models.gemma4_31b.model import (
            materialize_runtime_buffers,
        )

        mlx_source_transformations(model, dtype=torch.bfloat16)
        materialize_runtime_buffers(model, dtype=torch.bfloat16)

        # After source transforms: signature is (tokens, input_pos) → (B, 1, V)
        # Single-token decode
        tokens = torch.randint(0, TINY_CONFIG.vocab_size, (1, 1))
        input_pos = torch.tensor([0], dtype=torch.long)
        with torch.no_grad():
            out = model(tokens, input_pos)
        self.assertEqual(out.shape, torch.Size([1, TINY_CONFIG.vocab_size]))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

        # Multi-token prefill
        seq_len = 4
        tokens = torch.randint(0, TINY_CONFIG.vocab_size, (1, seq_len))
        input_pos = torch.arange(seq_len, dtype=torch.long)
        with torch.no_grad():
            out = model(tokens, input_pos)
        self.assertEqual(out.shape, torch.Size([1, TINY_CONFIG.vocab_size]))
        self.assertFalse(torch.isnan(out).any())

    def test_export_to_pte(self):
        """Full export: quantize → pack → export with MLXPartitioner."""
        try:
            from executorch.backends.mlx import MLXPartitioner  # noqa: F401
        except ImportError:
            self.skipTest("MLX backend not available")

        from executorch.examples.models.gemma4_31b.export import (
            export_and_lower,
            load_prequantized_model,
        )

        with tempfile.TemporaryDirectory() as ckpt_dir, tempfile.TemporaryDirectory() as out_dir:
            save_checkpoint(ckpt_dir)
            with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
                json.dump(config_dict(), f)

            model, config = load_prequantized_model(
                ckpt_dir, max_seq_len=TINY_CONFIG.max_seq_len, backend="mlx"
            )
            export_and_lower(model, config, out_dir, backend="mlx")
            self.assertTrue(os.path.exists(os.path.join(out_dir, "model.pte")))


if __name__ == "__main__":
    unittest.main()
