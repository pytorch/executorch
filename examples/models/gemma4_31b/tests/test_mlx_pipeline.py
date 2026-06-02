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
from executorch.examples.models.gemma4_31b.export import (
    _checkpoint_has_int8_vision_pe,
    _get_packers,
)

from executorch.examples.models.gemma4_31b.model import Gemma4_31B
from executorch.examples.models.gemma4_31b.pack_vision import (
    quantize_vision_position_table,
)
from executorch.examples.models.gemma4_31b.quant import (
    pack_model,
    QuantConfig,
    quantize_model,
    QuantRecipe,
    QuantRule,
)
from executorch.examples.models.gemma4_31b.tests.test_pipeline import (
    build_random_tiny_model,
    config_dict,
    DEFAULT_RECIPE,
    TINY_CONFIG,
)

# The MLX delegate (custom ops + generated serialization files) is only
# installed on macOS / the mlx CI image. On other hosts importing it raises, so
# the MLX-execution tests below are skipped (they run in
# .github/workflows/mlx.yml on macOS).
try:
    import executorch.backends.mlx.custom_ops  # noqa: F401

    _MLX_AVAILABLE = True
except Exception:
    _MLX_AVAILABLE = False

_requires_mlx = unittest.skipUnless(
    _MLX_AVAILABLE, "MLX delegate not installed (runs on macOS / mlx CI)"
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
        # Vision side stays bf16; the PE table is quantized explicitly before
        # calling quantize_model.
        QuantRule(r"vision_tower\..*", None),
        QuantRule(r"embed_vision\..*", None),
        QuantRule(r".*norm\.weight", None),
        QuantRule(r".*\.(v_proj|down_proj)\.weight", _INT8, layers=_EDGE_LAYERS),
        QuantRule(r".*\.weight", _INT4),
    ],
)


def save_vision_checkpoint(output_dir: str):
    """Save a tiny checkpoint matching quantize_and_save.py's vision format."""
    from safetensors.torch import save_file
    from torchao.prototype.safetensors.safetensors_support import (
        flatten_tensor_state_dict,
    )

    model = build_random_tiny_model()
    model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())

    quantize_vision_position_table(model.vision_tower)
    state_dict = quantize_model(model, DEFAULT_RECIPE)

    os.makedirs(output_dir, exist_ok=True)
    td, md = flatten_tensor_state_dict(state_dict)
    save_file(td, os.path.join(output_dir, "model.safetensors"), metadata=md)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict(), f)


class TestMlxPipeline(unittest.TestCase):
    """End-to-end: quantize -> pack for MLX -> forward."""

    def test_checkpoint_has_int8_vision_pe(self):
        """The MLX loader detects checkpoints with quantized vision PE buffers."""
        from safetensors.torch import save_file
        from torchao.prototype.safetensors.safetensors_support import (
            flatten_tensor_state_dict,
        )

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.safetensors")
            plain = {
                "vision_tower.patch_embedder._pet_int8": torch.zeros(
                    2, 4, 8, dtype=torch.int8
                ),
                "vision_tower.patch_embedder._pet_scale": torch.ones(
                    2, 4, 1, dtype=torch.float32
                ),
            }
            td, md = flatten_tensor_state_dict(plain)
            save_file(td, path, metadata=md)
            self.assertTrue(_checkpoint_has_int8_vision_pe(path))

    def test_checkpoint_has_int8_vision_pe_rejects_incomplete_pair(self):
        """The MLX loader rejects checkpoints with only one quantized PE buffer."""
        from safetensors.torch import save_file
        from torchao.prototype.safetensors.safetensors_support import (
            flatten_tensor_state_dict,
        )

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.safetensors")
            plain = {
                "vision_tower.patch_embedder._pet_int8": torch.zeros(
                    2, 4, 8, dtype=torch.int8
                ),
            }
            td, md = flatten_tensor_state_dict(plain)
            save_file(td, path, metadata=md)
            with self.assertRaisesRegex(RuntimeError, "Incomplete quantized vision"):
                _checkpoint_has_int8_vision_pe(path)

    def test_pack_for_mlx(self):
        """Quantize with sensitive recipe, pack for MLX, no meta weights."""
        model = build_random_tiny_model()
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
        quantize_vision_position_table(model.vision_tower)
        state_dict = quantize_model(model, TINY_SENSITIVE_RECIPE)

        with torch.device("meta"):
            model = Gemma4_31B(TINY_CONFIG)
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
        pack_model(model, state_dict, _get_packers("mlx"))

        for fqn, p in model.named_parameters():
            self.assertNotEqual(p.device.type, "meta", f"Weight '{fqn}' still on meta")

    def test_forward_after_pack(self):
        """Model produces valid output after MLX packing."""
        model = build_random_tiny_model()
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
        quantize_vision_position_table(model.vision_tower)
        state_dict = quantize_model(model, TINY_SENSITIVE_RECIPE)

        with torch.device("meta"):
            model = Gemma4_31B(TINY_CONFIG)
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
        pack_model(model, state_dict, _get_packers("mlx"))
        model.eval()

        from executorch.examples.models.gemma4_31b.model import (
            materialize_runtime_buffers,
        )

        materialize_runtime_buffers(model, dtype=torch.bfloat16)

        tokens = torch.randint(0, TINY_CONFIG.vocab_size, (1, 1))
        input_pos = torch.tensor([0], dtype=torch.long)
        temp = torch.tensor([1e-6], dtype=torch.float32)

        with torch.no_grad():
            out = model.decode_forward(tokens, input_pos, temp)

        self.assertEqual(out.shape, torch.Size([1, 1]))
        self.assertFalse(torch.isnan(out).any())

    def test_multi_token_forward(self):
        model = build_random_tiny_model()
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
        quantize_vision_position_table(model.vision_tower)
        state_dict = quantize_model(model, TINY_SENSITIVE_RECIPE)

        with torch.device("meta"):
            model = Gemma4_31B(TINY_CONFIG)
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
        pack_model(model, state_dict, _get_packers("mlx"))
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
            out = model.decode_forward(tokens, input_pos, temp)

        self.assertEqual(out.shape, torch.Size([1, 1]))
        self.assertFalse(torch.isnan(out).any())

    @_requires_mlx
    def test_source_transforms_forward(self):
        """Model produces valid output after MLX source transforms."""
        model = build_random_tiny_model()
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
        quantize_vision_position_table(model.vision_tower)
        state_dict = quantize_model(model, TINY_SENSITIVE_RECIPE)

        with torch.device("meta"):
            model = Gemma4_31B(TINY_CONFIG)
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
        pack_model(model, state_dict, _get_packers("mlx"))
        model.eval()

        from executorch.examples.models.gemma4_31b.mlx_source_transformations import (
            mlx_source_transformations,
        )
        from executorch.examples.models.gemma4_31b.model import (
            materialize_runtime_buffers,
        )

        mlx_source_transformations(model, dtype=torch.bfloat16)
        materialize_runtime_buffers(model, dtype=torch.bfloat16)

        # After source transforms: token-input forward returns logits.
        tokens = torch.randint(0, TINY_CONFIG.vocab_size, (1, 1))
        input_pos = torch.tensor([0], dtype=torch.long)
        with torch.no_grad():
            out = model(tokens, input_pos)
        self.assertEqual(out.shape, torch.Size([1, TINY_CONFIG.vocab_size]))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

        # Multi-token prefill: token-input forward must be identical to the new
        # exported path embed_text + mlx_prefill_forward.
        seq_len = 4
        tokens = torch.randint(0, TINY_CONFIG.vocab_size, (1, seq_len))
        input_pos = torch.arange(seq_len, dtype=torch.long)
        with torch.no_grad():
            old_tito = model(tokens, input_pos)
            via_embeds = model.mlx_prefill_forward(model.embed_text(tokens), input_pos)
        self.assertEqual(old_tito.shape, torch.Size([1, TINY_CONFIG.vocab_size]))
        self.assertFalse(torch.isnan(old_tito).any())
        self.assertTrue(torch.equal(old_tito, via_embeds))

    @_requires_mlx
    def test_sliding_window_wraparound_matches_reference(self):
        """Sliding MLX ring-buffer attention masks only the active window."""
        from executorch.examples.models.gemma4_31b.mlx_source_transformations import (
            mlx_source_transformations,
        )
        from executorch.examples.models.gemma4_31b.model import (
            materialize_runtime_buffers,
        )

        torch.manual_seed(0)
        model = build_random_tiny_model()
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
        quantize_vision_position_table(model.vision_tower)
        state_dict = quantize_model(model, TINY_SENSITIVE_RECIPE)

        with torch.device("meta"):
            model = Gemma4_31B(TINY_CONFIG)
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
        pack_model(model, state_dict, _get_packers("mlx"))
        model.eval()

        mlx_source_transformations(model, dtype=torch.bfloat16)
        materialize_runtime_buffers(model, dtype=torch.bfloat16)

        sliding_layer = next(
            layer for layer in model.layers if layer.self_attn.is_sliding
        )
        attn = sliding_layer.self_attn
        window = attn.kv_cache.window_size
        buffer_size = attn.kv_cache.buffer_size
        positions = [
            0,
            window - 1,
            window,
            buffer_size - 1,
            buffer_size,
            buffer_size + 1,
        ]

        for pos in positions:
            tokens = torch.randint(0, TINY_CONFIG.vocab_size, (1, 1))
            input_pos = torch.tensor([pos], dtype=torch.long)
            with torch.no_grad():
                out = model.mlx_prefill_forward(model.embed_text(tokens), input_pos)

            self.assertEqual(out.shape, torch.Size([1, TINY_CONFIG.vocab_size]))
            self.assertFalse(torch.isnan(out).any(), f"NaN at position {pos}")
            self.assertFalse(torch.isinf(out).any(), f"Inf at position {pos}")

        # The sliding cache path intentionally passes buffer_size - T to
        # custom_sdpa so the fake op slices the whole ring buffer; the additive
        # mask then selects the valid causal window even after wraparound.
        mask = attn.kv_cache.create_sliding_window_mask(buffer_size + 1, 1)[0, 0, 0]
        allowed_slots = torch.nonzero(mask == 0, as_tuple=False).flatten().tolist()
        expected_slots = [0, 1] + list(range(buffer_size - window + 2, buffer_size))
        self.assertEqual(allowed_slots, expected_slots)
        self.assertEqual(len(allowed_slots), window)

    @_requires_mlx
    def test_source_transforms_use_mlx_ops(self):
        """Verify the traced graph contains the expected MLX custom ops.

        Each attention layer should produce:
          - 2× ``mlx.rope`` (q and k)
          - 2× ``mlx.kv_cache_update`` (k and v)
          - 1× ``mlx.custom_sdpa``
        """
        from executorch.examples.models.gemma4_31b.mlx_source_transformations import (
            mlx_source_transformations,
        )
        from executorch.examples.models.gemma4_31b.model import (
            materialize_runtime_buffers,
        )
        from torch.export import Dim, export

        model = build_random_tiny_model()
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
        quantize_vision_position_table(model.vision_tower)
        state_dict = quantize_model(model, TINY_SENSITIVE_RECIPE)

        with torch.device("meta"):
            model = Gemma4_31B(TINY_CONFIG)
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
        pack_model(model, state_dict, _get_packers("mlx"))
        model.eval()

        mlx_source_transformations(model, dtype=torch.bfloat16)
        materialize_runtime_buffers(model, dtype=torch.bfloat16)

        # Trace with dynamic seq_len matching the MLX export shape.
        seq_dim = Dim("seq", min=1, max=8)
        ep = export(
            model,
            (torch.tensor([[1, 2]]), torch.tensor([0, 1])),
            dynamic_shapes=({1: seq_dim}, {0: seq_dim}),
            strict=True,
        )

        op_counts = {"rope": 0, "kv_cache_update": 0, "custom_sdpa": 0}
        for node in ep.graph.nodes:
            if node.op != "call_function":
                continue
            name = str(node.target)
            for op in op_counts:
                if f"mlx.{op}" in name:
                    op_counts[op] += 1

        n_layers = TINY_CONFIG.num_hidden_layers
        self.assertEqual(op_counts["rope"], 2 * n_layers, f"got {op_counts}")
        self.assertEqual(op_counts["kv_cache_update"], 2 * n_layers, f"got {op_counts}")
        self.assertEqual(op_counts["custom_sdpa"], n_layers, f"got {op_counts}")

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
            save_vision_checkpoint(ckpt_dir)
            with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
                json.dump(config_dict(), f)

            model, config = load_prequantized_model(
                ckpt_dir, max_seq_len=TINY_CONFIG.max_seq_len, backend="mlx"
            )
            export_and_lower(model, config, out_dir, backend="mlx")
            self.assertTrue(os.path.exists(os.path.join(out_dir, "model.pte")))


class TestGgufMlxPipeline(unittest.TestCase):
    """Test GGUF → MLX loading path with synthetic Q6_K-like tensors."""

    @unittest.skip(
        "GGUF vision model loading lands in the g4-vision-llama-cpp branch; "
        "this branch's gguf_loader is still the text-only one, incompatible "
        "with the multimodal-by-default model."
    )
    def test_load_gguf_model_mlx_backend(self):
        """gguf_loader.load_gguf_model accepts backend='mlx'."""
        try:
            import gguf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("gguf package not installed")

        from executorch.examples.models.gemma4_31b.gguf_loader import load_gguf_model

        # Will fail on missing file, but NOT on "Unsupported backend".
        with self.assertRaisesRegex((FileNotFoundError, OSError, RuntimeError), ".*"):
            load_gguf_model("/nonexistent.gguf", backend="mlx")

    def test_mlx_backend_rejects_unknown(self):
        from executorch.examples.models.gemma4_31b.gguf_loader import load_gguf_model

        with self.assertRaisesRegex(ValueError, "Unsupported backend"):
            load_gguf_model("/nonexistent.gguf", backend="tpu")

    def test_gs16_packing_preserves_values(self):
        """Q6_K-like weight (gs=16) preserves dequantized values after packing."""
        from executorch.examples.models.gemma4_31b.quant.pack_mlx import pack_for_mlx
        from executorch.examples.models.gemma4_31b.quant.quantize import (
            dequantize_weight,
        )
        from torchao.quantization import IntxUnpackedToInt8Tensor

        w = IntxUnpackedToInt8Tensor(
            qdata=torch.randint(-32, 31, (64, 128), dtype=torch.int8),
            scale=torch.randn(64, 8, dtype=torch.bfloat16),
            zero_point=torch.zeros(64, 8, dtype=torch.int8),
            target_dtype=torch.int8,
            block_size=(1, 16),
            dtype=torch.bfloat16,
            activation_quantization=None,
        )
        before = dequantize_weight(w, torch.float32)

        module = nn.Linear(128, 64, bias=False)
        pack_for_mlx(module, {"weight": w})
        after = dequantize_weight(module.weight.data, torch.float32)

        self.assertTrue(
            torch.allclose(before, after, atol=1e-5),
            f"max diff: {(before - after).abs().max():.6g}",
        )

    def test_embedding_packing_preserves_values(self):
        """MLX embedding packing preserves dequantized weight values."""
        from executorch.examples.models.gemma4_31b.quant.pack_mlx import pack_for_mlx
        from executorch.examples.models.gemma4_31b.quant.quantize import (
            dequantize_weight,
        )
        from torchao.quantization import IntxUnpackedToInt8Tensor

        w = IntxUnpackedToInt8Tensor(
            qdata=torch.randint(-8, 7, (256, 128), dtype=torch.int8),
            scale=torch.randn(256, 4, dtype=torch.bfloat16),
            zero_point=torch.zeros(256, 4, dtype=torch.bfloat16),
            target_dtype=torch.int4,
            block_size=(1, 32),
            dtype=torch.bfloat16,
            activation_quantization=None,
        )
        before = dequantize_weight(w, torch.float32)

        module = nn.Embedding(256, 128)
        pack_for_mlx(module, {"weight": w})
        after = dequantize_weight(module.weight.data, torch.float32)

        self.assertTrue(
            torch.allclose(before, after, atol=1e-5),
            f"max diff: {(before - after).abs().max():.6g}",
        )


if __name__ == "__main__":
    unittest.main()
