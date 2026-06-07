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
        state_dict = quantize_model(model, TINY_SENSITIVE_RECIPE)

        with torch.device("meta"):
            model = Gemma4_31B(TINY_CONFIG)
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
        pack_model(model, state_dict, DEFAULT_MLX_PACKERS)
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
            save_checkpoint(ckpt_dir)
            with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
                json.dump(config_dict(), f)

            model, config = load_prequantized_model(
                ckpt_dir, max_seq_len=TINY_CONFIG.max_seq_len, backend="mlx"
            )
            export_and_lower(model, config, out_dir, backend="mlx")
            self.assertTrue(os.path.exists(os.path.join(out_dir, "model.pte")))


class TestGgufMlxPipeline(unittest.TestCase):
    """Test GGUF → MLX loading path with synthetic Q6_K-like tensors."""

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


class TestGgufLinearMlx(unittest.TestCase):
    """GGUF-quantized linears (Q6_K + Q4_K) lower through the MLX GGUF pattern."""

    def _linear(self, N: int, K: int, ggml_type: str) -> nn.Module:
        from executorch.backends.mlx.custom_kernel_ops.gguf.test.test_linear import (
            make_q4_k_blob,
            make_q6_k_blob,
        )
        from executorch.extension.llm.export.gguf import ExportableGGUFTensor

        blob = (make_q6_k_blob if ggml_type == "q6_k" else make_q4_k_blob)(N, K)
        lin = nn.Linear(K, N, bias=False).to(torch.bfloat16)
        lin.weight = nn.Parameter(
            ExportableGGUFTensor.from_raw(blob, ggml_type, torch.bfloat16),
            requires_grad=False,
        )
        return lin.eval()

    def _assert_delegated(self, model, example, leftovers):
        import executorch.backends.mlx.custom_kernel_ops.gguf.patterns  # noqa: F401
        from executorch.backends.mlx import MLXPartitioner
        from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
        from torch.export import Dim, export

        seq = Dim("seq", min=1, max=8)
        ep = export(model, example, dynamic_shapes=({0: seq},), strict=True)
        et = to_edge_transform_and_lower(
            ep,
            partitioner=[MLXPartitioner()],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        remaining = [
            str(n.target)
            for n in et.exported_program().graph.nodes
            if n.op == "call_function" and any(t in str(n.target) for t in leftovers)
        ]
        self.assertEqual(remaining, [], f"not delegated to MLX: {remaining}")

    def test_q6k_linear_delegates(self):
        self._assert_delegated(
            self._linear(256, 512, "q6_k"),
            (torch.randn(4, 512, dtype=torch.bfloat16),),
            ("gguf_dequantize", "linear"),
        )

    def test_q4k_linear_delegates(self):
        self._assert_delegated(
            self._linear(512, 512, "q4_k"),
            (torch.randn(4, 512, dtype=torch.bfloat16),),
            ("gguf_dequantize", "linear"),
        )


class TestGgufEmbeddingMlx(unittest.TestCase):
    """GGUF token embeddings (Q6_K + Q4_K) lower through the MLX GGUF pattern."""

    def _assert_delegated(self, ggml_type: str):
        import executorch.backends.mlx.custom_kernel_ops.gguf.patterns  # noqa: F401
        from executorch.backends.mlx import MLXPartitioner
        from executorch.backends.mlx.custom_kernel_ops.gguf.test.test_linear import (
            make_q4_k_blob,
            make_q6_k_blob,
        )
        from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
        from executorch.extension.llm.export.gguf import ExportableGGUFTensor
        from torch.export import Dim, export

        vocab, K = 512, 256
        blob = (make_q6_k_blob if ggml_type == "q6_k" else make_q4_k_blob)(vocab, K)
        emb = nn.Embedding(vocab, K)
        emb.weight = nn.Parameter(
            ExportableGGUFTensor.from_raw(blob, ggml_type, torch.bfloat16),
            requires_grad=False,
        )
        emb = emb.eval()
        seq = Dim("seq", min=1, max=8)
        ep = export(
            emb,
            (torch.randint(0, vocab, (4,), dtype=torch.int64),),
            dynamic_shapes=({0: seq},),
            strict=True,
        )
        et = to_edge_transform_and_lower(
            ep,
            partitioner=[MLXPartitioner()],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        remaining = [
            str(n.target)
            for n in et.exported_program().graph.nodes
            if n.op == "call_function"
            and any(t in str(n.target) for t in ("gguf_dequantize", "embedding"))
        ]
        self.assertEqual(remaining, [], f"not delegated to MLX: {remaining}")

    def test_q6k_embedding_delegates(self):
        self._assert_delegated("q6_k")

    def test_q4k_embedding_delegates(self):
        self._assert_delegated("q4_k")


if __name__ == "__main__":
    unittest.main()
