# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for FuseInt4WeightOnlyQuantMatmulPass.

Tests the fusion pass that combines multiple int4pack_mm operations sharing
the same input into a single fused operation.
"""

import unittest
from typing import Tuple

import torch
from executorch.backends.cuda.passes import FuseInt4WeightOnlyQuantMatmulPass
from executorch.exir.program._program import _update_exported_program_graph_module
from torch.export import Dim, export
from torch.fx import GraphModule
from torch.fx.passes.dialect.common.cse_pass import CSEPass
from torch.fx.passes.infra.pass_base import PassResult


class TestFuseInt4QuantMatmul(unittest.TestCase):
    """Test FuseInt4WeightOnlyQuantMatmulPass public interface."""

    def setUp(self):
        """Set up test environment."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        # Check if torchao is available
        try:
            from torchao.quantization.quantize_.workflows.int4.int4_tile_packed_to_4d_tensor import (
                Int4TilePackedTo4dTensor,
            )
            from torchao.quantization.utils import pack_tinygemm_scales_and_zeros
            from torchao.utils import find_multiple
        except ImportError:
            self.skipTest("torchao is not available")

    def _create_int4_weight(
        self, out_features: int, in_features: int, block_size: int = 128
    ):
        """
        Create Int4TilePackedTo4dTensor for testing.

        This creates a proper quantized weight tensor that will use
        torch.ops.aten._weight_int4pack_mm when used in nn.functional.linear.
        """
        from torchao.quantization.quantize_.workflows.int4.int4_tile_packed_to_4d_tensor import (
            Int4TilePackedTo4dTensor,
        )
        from torchao.quantization.utils import pack_tinygemm_scales_and_zeros
        from torchao.utils import find_multiple

        device = "cuda"
        inner_k_tiles = 8

        # Pad dimensions to required multiples
        in_features_padded = find_multiple(in_features, 1024)
        out_features_padded = find_multiple(out_features, 8)

        # Create INT4 values in range [-8, 7]
        int4_values = torch.randint(
            -8, 8, (out_features, in_features), dtype=torch.int8, device=device
        )

        # Create scales
        num_blocks = in_features // block_size
        scales = (
            torch.randn(out_features, num_blocks, dtype=torch.bfloat16, device=device)
            * 0.01
        )

        # Pad int4 values
        int4_padded = torch.nn.functional.pad(
            int4_values,
            (
                0,
                in_features_padded - in_features,
                0,
                out_features_padded - out_features,
            ),
            value=0,
        )

        # Pad scales
        num_blocks_padded = in_features_padded // block_size
        scales_padded = torch.nn.functional.pad(
            scales,
            (0, num_blocks_padded - num_blocks, 0, out_features_padded - out_features),
            value=1.0,
        )

        # Convert to unsigned [0, 15]
        int4_shifted = (int4_padded + 8).to(torch.int32).clamp(0, 15)

        # Pack two INT4 values per uint8
        int_data_packed = (int4_shifted[:, ::2] << 4 | int4_shifted[:, 1::2]).to(
            torch.uint8
        )

        # Convert to tinygemm format
        packed_weight = torch.ops.aten._convert_weight_to_int4pack(
            int_data_packed.contiguous(), inner_k_tiles
        )

        # Create scale_and_zero
        zero_points = torch.zeros_like(scales_padded, dtype=scales_padded.dtype)
        scale_and_zero = pack_tinygemm_scales_and_zeros(
            scales_padded.reshape(out_features_padded, -1),
            zero_points.reshape(out_features_padded, -1),
            scales.dtype,
        )

        return Int4TilePackedTo4dTensor(
            qdata=packed_weight,
            scale_and_zero=scale_and_zero,
            block_size=[1, block_size],
            shape=(out_features, in_features),
            act_pre_scale=None,
        )

    def _apply_fusion_pipeline(self, exported_program, min_fusion_size=2, max_fusion_size=3):
        """
        Apply CSE + Fusion passes matching cuda_backend.py preprocessing.

        This tests the integration of CSEPass and FuseInt4WeightOnlyQuantMatmulPass
        using the same pattern as CudaBackend.preprocess().
        """
        # CSE pass (required to create fuseable patterns)
        cse_pass = CSEPass()
        cse_result = cse_pass(exported_program.graph_module)
        if cse_result.modified:
            exported_program = _update_exported_program_graph_module(
                exported_program, cse_result.graph_module
            )

        # Fusion pass
        fusion_pass = FuseInt4WeightOnlyQuantMatmulPass(
            min_fusion_size=min_fusion_size, max_fusion_size=max_fusion_size
        )
        fusion_result = fusion_pass(exported_program.graph_module)
        if fusion_result.modified:
            exported_program = _update_exported_program_graph_module(
                exported_program, fusion_result.graph_module
            )

        return exported_program

    def _count_int4mm_ops(self, graph_module: GraphModule) -> int:
        """Count number of _weight_int4pack_mm operations in graph.

        Handles both standard torch ops and EdgeOpOverload (edge dialect).
        """
        count = 0
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            target = node.target
            # Direct match for standard torch op
            if target == torch.ops.aten._weight_int4pack_mm.default:
                count += 1
            # Handle EdgeOpOverload (edge dialect wraps ops)
            elif "_weight_int4pack_mm" in str(target):
                count += 1
        return count

    def _build_qkv_model(
        self, hidden_dim: int = 2048, block_size: int = 128
    ) -> torch.nn.Module:
        """
        Build a model with Q/K/V projections pattern (3 int4mm ops sharing input).

        This simulates attention projection layers that should be fused 3→1.
        """

        class QKVModel(torch.nn.Module):
            def __init__(self, create_weight_fn):
                super().__init__()
                # Create linear layers with Int4TilePackedTo4dTensor weights
                self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False, device="cuda")
                self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False, device="cuda")
                self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False, device="cuda")

                # Replace weights with INT4 quantized tensors
                self.q_proj.weight = torch.nn.Parameter(
                    create_weight_fn(hidden_dim, hidden_dim, block_size), requires_grad=False
                )
                self.k_proj.weight = torch.nn.Parameter(
                    create_weight_fn(hidden_dim, hidden_dim, block_size), requires_grad=False
                )
                self.v_proj.weight = torch.nn.Parameter(
                    create_weight_fn(hidden_dim, hidden_dim, block_size), requires_grad=False
                )

            def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                # Three int4mm operations sharing the same input
                q = torch.nn.functional.linear(x, self.q_proj.weight)
                k = torch.nn.functional.linear(x, self.k_proj.weight)
                v = torch.nn.functional.linear(x, self.v_proj.weight)
                return q, k, v

        return QKVModel(self._create_int4_weight).eval()

    def _build_gate_up_model(
        self, hidden_dim: int = 2048, intermediate_dim: int = 8192, block_size: int = 128
    ) -> torch.nn.Module:
        """
        Build a model with Gate/Up projection pattern (2 int4mm ops sharing input).

        This simulates MLP layers that should be fused 2→1.
        """

        class GateUpModel(torch.nn.Module):
            def __init__(self, create_weight_fn):
                super().__init__()
                # Create linear layers with Int4TilePackedTo4dTensor weights
                self.gate_proj = torch.nn.Linear(
                    hidden_dim, intermediate_dim, bias=False, device="cuda"
                )
                self.up_proj = torch.nn.Linear(
                    hidden_dim, intermediate_dim, bias=False, device="cuda"
                )

                # Replace weights with INT4 quantized tensors
                self.gate_proj.weight = torch.nn.Parameter(
                    create_weight_fn(intermediate_dim, hidden_dim, block_size),
                    requires_grad=False,
                )
                self.up_proj.weight = torch.nn.Parameter(
                    create_weight_fn(intermediate_dim, hidden_dim, block_size),
                    requires_grad=False,
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Two int4mm operations sharing the same input
                gate = torch.nn.functional.linear(x, self.gate_proj.weight)
                up = torch.nn.functional.linear(x, self.up_proj.weight)
                return gate * up

        return GateUpModel(self._create_int4_weight).eval()

    def _build_different_inputs_model(
        self, hidden_dim: int = 2048, block_size: int = 128
    ) -> torch.nn.Module:
        """
        Build a model with operations on different inputs (should NOT fuse).
        """

        class DifferentInputsModel(torch.nn.Module):
            def __init__(self, create_weight_fn):
                super().__init__()
                # Create linear layers with Int4TilePackedTo4dTensor weights
                self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False, device="cuda")
                self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False, device="cuda")

                # Replace weights with INT4 quantized tensors
                self.linear1.weight = torch.nn.Parameter(
                    create_weight_fn(hidden_dim, hidden_dim, block_size), requires_grad=False
                )
                self.linear2.weight = torch.nn.Parameter(
                    create_weight_fn(hidden_dim, hidden_dim, block_size), requires_grad=False
                )

            def forward(
                self, x: torch.Tensor, y: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                # Two operations with different inputs - should NOT fuse
                out1 = torch.nn.functional.linear(x, self.linear1.weight)
                out2 = torch.nn.functional.linear(y, self.linear2.weight)
                return out1, out2

        return DifferentInputsModel(self._create_int4_weight).eval()

    def _build_cross_attention_model(
        self, decoder_dim: int = 1280, encoder_dim: int = 1280, block_size: int = 128
    ) -> torch.nn.Module:
        """
        Build a model with cross-attention pattern (Whisper-like).

        Pattern: Q from decoder, K/V from encoder (K/V should fuse 2→1, Q separate).
        This simulates Whisper decoder cross-attention where K/V share encoder
        output but Q uses decoder hidden state.
        """

        class CrossAttentionModel(torch.nn.Module):
            def __init__(self, create_weight_fn):
                super().__init__()
                # Q projection from decoder hidden state
                self.q_proj = torch.nn.Linear(decoder_dim, decoder_dim, bias=False, device="cuda")
                # K/V projections from encoder output
                self.k_proj = torch.nn.Linear(encoder_dim, decoder_dim, bias=False, device="cuda")
                self.v_proj = torch.nn.Linear(encoder_dim, decoder_dim, bias=False, device="cuda")

                # Replace weights with INT4 quantized tensors
                self.q_proj.weight = torch.nn.Parameter(
                    create_weight_fn(decoder_dim, decoder_dim, block_size), requires_grad=False
                )
                self.k_proj.weight = torch.nn.Parameter(
                    create_weight_fn(decoder_dim, encoder_dim, block_size), requires_grad=False
                )
                self.v_proj.weight = torch.nn.Parameter(
                    create_weight_fn(decoder_dim, encoder_dim, block_size), requires_grad=False
                )

            def forward(
                self, decoder_hidden: torch.Tensor, encoder_output: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                # Cross-attention: Q from decoder, K/V from encoder
                q = torch.nn.functional.linear(decoder_hidden, self.q_proj.weight)
                k = torch.nn.functional.linear(encoder_output, self.k_proj.weight)
                v = torch.nn.functional.linear(encoder_output, self.v_proj.weight)
                return q, k, v

        return CrossAttentionModel(self._create_int4_weight).eval()

    def _build_sequential_model(
        self, hidden_dim: int = 2048, intermediate_dim: int = 8192, block_size: int = 128
    ) -> torch.nn.Module:
        """
        Build a model with sequential operations (should NOT fuse).

        This simulates Whisper MLP: fc1 → GELU → fc2 (sequential chain).
        """

        class SequentialModel(torch.nn.Module):
            def __init__(self, create_weight_fn):
                super().__init__()
                # Sequential MLP layers
                self.fc1 = torch.nn.Linear(
                    hidden_dim, intermediate_dim, bias=False, device="cuda"
                )
                self.fc2 = torch.nn.Linear(
                    intermediate_dim, hidden_dim, bias=False, device="cuda"
                )

                # Replace weights with INT4 quantized tensors
                self.fc1.weight = torch.nn.Parameter(
                    create_weight_fn(intermediate_dim, hidden_dim, block_size),
                    requires_grad=False,
                )
                self.fc2.weight = torch.nn.Parameter(
                    create_weight_fn(hidden_dim, intermediate_dim, block_size),
                    requires_grad=False,
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Sequential operations: fc2 depends on fc1 output
                x = torch.nn.functional.linear(x, self.fc1.weight)
                x = torch.nn.functional.gelu(x)
                x = torch.nn.functional.linear(x, self.fc2.weight)
                return x

        return SequentialModel(self._create_int4_weight).eval()

    def test_fuse_qkv_projection(self):
        """Test fusion of Q/K/V projections (3→1 operation)."""
        model = self._build_qkv_model(hidden_dim=2048, block_size=128)
        example_input = (torch.randn(4, 128, 2048, device="cuda", dtype=torch.bfloat16),)

        # Export model
        exported = export(model, example_input, strict=False)
        ops_before = self._count_int4mm_ops(exported.graph_module)
        self.assertEqual(ops_before, 3, "Expected 3 int4mm operations before fusion")

        # Apply CSE + Fusion pipeline
        result = self._apply_fusion_pipeline(exported)

        # Verify fusion: 3 → 1
        ops_after = self._count_int4mm_ops(result.graph_module)
        self.assertEqual(ops_after, 1, "Expected 1 int4mm operation after fusion (3→1)")
        result.graph_module.graph.lint()

    def test_fuse_gate_up_projection(self):
        """Test fusion of Gate/Up projections (2→1 operation)."""
        model = self._build_gate_up_model(
            hidden_dim=2048, intermediate_dim=8192, block_size=128
        )
        example_input = (torch.randn(4, 128, 2048, device="cuda", dtype=torch.bfloat16),)

        # Export model
        exported = export(model, example_input, strict=False)
        ops_before = self._count_int4mm_ops(exported.graph_module)
        self.assertEqual(ops_before, 2, "Expected 2 int4mm operations before fusion")

        # Apply CSE + Fusion pipeline
        result = self._apply_fusion_pipeline(exported)

        # Verify fusion: 2 → 1
        ops_after = self._count_int4mm_ops(result.graph_module)
        self.assertEqual(ops_after, 1, "Expected 1 int4mm operation after fusion (2→1)")
        result.graph_module.graph.lint()

    def test_no_fusion_different_inputs(self):
        """Test that operations with different inputs are NOT fused."""
        model = self._build_different_inputs_model(hidden_dim=2048, block_size=128)
        example_inputs = (
            torch.randn(4, 128, 2048, device="cuda", dtype=torch.bfloat16),
            torch.randn(4, 128, 2048, device="cuda", dtype=torch.bfloat16),
        )

        # Export model
        exported = export(model, example_inputs, strict=False)
        ops_before = self._count_int4mm_ops(exported.graph_module)
        self.assertEqual(ops_before, 2, "Expected 2 int4mm operations before fusion")

        # Apply CSE + Fusion pipeline
        result = self._apply_fusion_pipeline(exported)

        # Verify NO fusion (different inputs)
        ops_after = self._count_int4mm_ops(result.graph_module)
        self.assertEqual(ops_after, 2, "Expected 2 int4mm operations (unchanged)")

    def test_respects_min_fusion_size(self):
        """Test that fusion respects min_fusion_size parameter."""
        model = self._build_qkv_model(hidden_dim=2048, block_size=128)
        example_input = (torch.randn(4, 128, 2048, device="cuda", dtype=torch.bfloat16),)

        # Export model
        exported = export(model, example_input, strict=False)

        # Apply pipeline with min_fusion_size=4 (should NOT fuse 3 ops)
        result = self._apply_fusion_pipeline(exported, min_fusion_size=4, max_fusion_size=5)

        # Verify NO fusion (below min_fusion_size)
        ops_after = self._count_int4mm_ops(result.graph_module)
        self.assertEqual(ops_after, 3, "Expected 3 int4mm operations (unchanged)")

    def test_dynamic_shapes_with_symint(self):
        """
        Test fusion with dynamic shapes (SymInt).

        Critical for models with dynamic sequence lengths (Voxtral, other LLMs).
        """
        model = self._build_qkv_model(hidden_dim=3072, block_size=128)

        # Export with dynamic shapes
        seq_length = 3
        inputs_embeds = torch.randn(1, seq_length, 3072, device="cuda", dtype=torch.bfloat16)
        seq_len_dim = Dim("seq_length_dim", max=128)
        dynamic_shapes = ({1: seq_len_dim},)

        exported = export(model, (inputs_embeds,), dynamic_shapes=dynamic_shapes, strict=False)
        ops_before = self._count_int4mm_ops(exported.graph_module)
        self.assertEqual(ops_before, 3, "Expected 3 int4mm operations before fusion")

        # Apply CSE + Fusion pipeline (should handle SymInt correctly with device='meta')
        result = self._apply_fusion_pipeline(exported)

        # Verify fusion works with dynamic shapes
        ops_after = self._count_int4mm_ops(result.graph_module)
        self.assertEqual(
            ops_after, 1, "Expected 1 int4mm operation after fusion with dynamic shapes"
        )
        result.graph_module.graph.lint()

    def test_preserves_graph_validity(self):
        """Test that fusion preserves graph validity and metadata."""
        model = self._build_qkv_model(hidden_dim=2048, block_size=128)
        example_input = (torch.randn(4, 128, 2048, device="cuda", dtype=torch.bfloat16),)

        # Export and apply fusion
        exported = export(model, example_input, strict=False)
        result = self._apply_fusion_pipeline(exported)

        # Verify graph validity
        result.graph_module.graph.lint()  # Should not raise
        code = result.graph_module.code
        self.assertIsNotNone(code, "Fused graph should generate code")

        # Verify all function nodes have targets
        for node in result.graph_module.graph.nodes:
            if node.op == "call_function":
                self.assertIsNotNone(node.target, f"Node {node.name} missing target")

    def test_fuse_cross_attention_kv(self):
        """
        Test fusion of cross-attention K/V projections (Whisper pattern).

        Cross-attention pattern: Q from decoder, K/V from encoder.
        Expected: K/V fuse (2→1), Q stays separate (total 3→2).
        """
        model = self._build_cross_attention_model(
            decoder_dim=1280, encoder_dim=1280, block_size=128
        )
        decoder_hidden = torch.randn(4, 128, 1280, device="cuda", dtype=torch.bfloat16)
        encoder_output = torch.randn(4, 256, 1280, device="cuda", dtype=torch.bfloat16)
        example_inputs = (decoder_hidden, encoder_output)

        # Export model
        exported = export(model, example_inputs, strict=False)
        ops_before = self._count_int4mm_ops(exported.graph_module)
        self.assertEqual(ops_before, 3, "Expected 3 int4mm operations before fusion")

        # Apply CSE + Fusion pipeline
        result = self._apply_fusion_pipeline(exported)

        # Verify partial fusion: K/V fuse (2→1), Q separate = 2 ops total
        ops_after = self._count_int4mm_ops(result.graph_module)
        self.assertEqual(
            ops_after, 2, "Expected 2 int4mm operations after fusion (K/V fused, Q separate)"
        )
        result.graph_module.graph.lint()

    def test_zero_copy_split_no_materialization(self):
        """
        Test that fused QKV split does not introduce memory copies.

        After fusion, the split operation should create views (zero-copy) rather
        than materialized copies. This test verifies that:
        1. Split operations exist in the graph (tensor_split or slice)
        2. NO contiguous/clone/copy operations are inserted after split
        3. The graph structure supports zero-copy views

        This is critical for performance: materialization adds 29% overhead,
        while zero-copy split provides 17% speedup (per roofline analysis).
        """
        model = self._build_qkv_model(hidden_dim=1280, block_size=128)
        example_input = (torch.randn(1, 128, 1280, device="cuda", dtype=torch.bfloat16),)

        # Export and apply fusion
        exported = export(model, example_input, strict=False)
        result = self._apply_fusion_pipeline(exported)

        # Verify fusion occurred (3→1)
        ops_after = self._count_int4mm_ops(result.graph_module)
        self.assertEqual(ops_after, 1, "Expected 1 fused int4mm operation")

        # Track split operations and potential materializations
        has_tensor_split = False
        has_slice_ops = False
        materialization_ops = []
        split_output_nodes = set()
        fused_int4mm_node = None

        # First pass: identify fused int4mm and split operations
        for node in result.graph_module.graph.nodes:
            if node.op == "call_function":
                # Find the fused int4mm node
                if node.target == torch.ops.aten._weight_int4pack_mm.default:
                    fused_int4mm_node = node

                # Check for split operations
                if "tensor_split" in str(node.target):
                    has_tensor_split = True
                    split_output_nodes.add(node)
                    # All users of tensor_split are getitem nodes
                    for user in node.users:
                        split_output_nodes.add(user)
                elif node.target == torch.ops.aten.slice.Tensor:
                    has_slice_ops = True
                    split_output_nodes.add(node)

        # The fused int4mm output is used by either tensor_split or slice operations
        # Check the users of the fused int4mm node
        if fused_int4mm_node:
            for user in fused_int4mm_node.users:
                target_str = str(user.target) if user.op == "call_function" else ""
                if "tensor_split" in target_str or "split" in target_str:
                    has_tensor_split = True
                    split_output_nodes.add(user)
                    # Add getitem users
                    for getitem_user in user.users:
                        split_output_nodes.add(getitem_user)
                elif user.target == torch.ops.aten.slice.Tensor:
                    has_slice_ops = True
                    split_output_nodes.add(user)

        # Verify we have split operations
        self.assertTrue(
            has_tensor_split or has_slice_ops,
            "Expected tensor_split or slice operations in fused graph"
        )

        # Second pass: check for materialization operations AFTER split
        # These would indicate forced memory copies
        for node in result.graph_module.graph.nodes:
            if node.op == "call_function":
                # Check if this op acts on split outputs
                is_downstream_of_split = any(
                    arg in split_output_nodes
                    for arg in node.args
                    if isinstance(arg, torch.fx.Node)
                )

                if is_downstream_of_split:
                    # Check for operations that force materialization
                    target_str = str(node.target)
                    if any(
                        op in target_str.lower()
                        for op in ["contiguous", "clone", "copy", "_copy"]
                    ):
                        materialization_ops.append((node.name, target_str))

        # Assert no forced materializations
        self.assertEqual(
            len(materialization_ops),
            0,
            f"Found materialization operations after split: {materialization_ops}\n"
            f"This indicates the split is NOT zero-copy and will hurt performance.\n"
            f"Expected: Views only (tensor_split or slice)\n"
            f"Found: {materialization_ops}",
        )

        # Verify graph structure
        result.graph_module.graph.lint()

        # Additional verification: count split-related ops
        split_op_count = sum(
            1
            for node in result.graph_module.graph.nodes
            if node.op == "call_function"
            and (
                "tensor_split" in str(node.target)
                or node.target == torch.ops.aten.slice.Tensor
            )
        )

        if has_tensor_split:
            # tensor_split: 1 split node + 3 getitem nodes
            getitem_count = sum(
                1
                for node in result.graph_module.graph.nodes
                if node.op == "call_function" and "getitem" in str(node.target)
            )
            # We should have exactly 1 tensor_split
            self.assertGreaterEqual(split_op_count, 1, "Expected at least 1 tensor_split operation")
            # And 3 getitem nodes (Q, K, V)
            self.assertEqual(getitem_count, 3, "Expected 3 getitem operations for Q/K/V")
        else:
            # Explicit slicing: 3 slice operations (one each for Q, K, V)
            self.assertEqual(split_op_count, 3, "Expected 3 slice operations for Q/K/V")

    def test_no_fusion_sequential_ops(self):
        """
        Test that sequential operations do NOT fuse (Whisper MLP pattern).

        Sequential pattern: fc1 → GELU → fc2 (fc2 depends on fc1 output).
        Expected: 2→2 (no fusion, different inputs).
        """
        model = self._build_sequential_model(
            hidden_dim=1280, intermediate_dim=5120, block_size=128
        )
        example_input = (torch.randn(4, 128, 1280, device="cuda", dtype=torch.bfloat16),)

        # Export model
        exported = export(model, example_input, strict=False)
        ops_before = self._count_int4mm_ops(exported.graph_module)
        self.assertEqual(ops_before, 2, "Expected 2 int4mm operations before fusion")

        # Apply CSE + Fusion pipeline
        result = self._apply_fusion_pipeline(exported)

        # Verify NO fusion (sequential operations have different inputs)
        ops_after = self._count_int4mm_ops(result.graph_module)
        self.assertEqual(
            ops_after, 2, "Expected 2 int4mm operations (no fusion for sequential ops)"
        )
        result.graph_module.graph.lint()

    def test_aoti_backend_pass_application(self):
        """
        Regression test for AotiBackend.preprocess pass application.

        Tests that CudaBackend.get_custom_passes() returns passes that work
        correctly with the AotiBackend.preprocess pass application loop,
        specifically testing that PassResult is properly handled and
        _update_exported_program_graph_module is called when passes modify
        the graph.

        This test exercises the exact same code path as AotiBackend.preprocess()
        lines 158-166, ensuring that:
        1. get_custom_passes() returns CSEPass and FuseInt4WeightOnlyQuantMatmulPass
        2. Passes returning PassResult with modified=True trigger graph update
        3. The exported program is properly updated after each pass
        4. Fusion actually occurs (int4mm ops are reduced)
        """
        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.exir.backend.compile_spec_schema import CompileSpec

        # Build model with fuseable Q/K/V pattern
        model = self._build_qkv_model(hidden_dim=2048, block_size=128)
        example_input = (torch.randn(4, 128, 2048, device="cuda", dtype=torch.bfloat16),)

        # Export model
        exported = export(model, example_input, strict=False)
        ops_before = self._count_int4mm_ops(exported.graph_module)
        self.assertEqual(ops_before, 3, "Expected 3 int4mm operations before fusion")

        # Get custom passes from CudaBackend (same as preprocess does)
        compile_specs = [
            CompileSpec("triton_kernel_mode", b"OFF"),  # Disable Triton to simplify test
        ]
        custom_passes = CudaBackend.get_custom_passes(compile_specs)

        # Verify we get CSE and Fusion passes
        pass_types = [type(p).__name__ for p in custom_passes]
        self.assertIn("CSEPass", pass_types, "Expected CSEPass in custom passes")
        self.assertIn(
            "FuseInt4WeightOnlyQuantMatmulPass",
            pass_types,
            "Expected FuseInt4WeightOnlyQuantMatmulPass in custom passes",
        )

        # Apply passes using the EXACT same logic as AotiBackend.preprocess()
        # This is the critical code path we're testing (lines 158-168 of aoti_backend.py)
        device_edge_program = exported
        for custom_pass in custom_passes:
            result = custom_pass(device_edge_program.graph_module)
            # Handle passes that return PassResult with a new graph_module
            if isinstance(result, PassResult) and result.modified:
                # Use a permissive verifier that allows all operator types
                from torch._export.verifier import Verifier
                from torch._library.custom_ops import CustomOpDef
                from torch._ops import HigherOrderOperator, OpOverload, OpOverloadPacket

                class _PermissiveVerifier(Verifier):
                    dialect = "EDGE"

                    def allowed_op_types(self):
                        return (OpOverload, OpOverloadPacket, HigherOrderOperator, CustomOpDef)

                    def check_valid_op(self, op):
                        pass  # Allow all ops

                device_edge_program = _update_exported_program_graph_module(
                    device_edge_program, result.graph_module, override_verifiers=[_PermissiveVerifier]
                )

        # Verify fusion occurred (3→1)
        ops_after = self._count_int4mm_ops(device_edge_program.graph_module)
        self.assertEqual(
            ops_after,
            1,
            f"Expected 1 int4mm operation after fusion via AotiBackend pass application, "
            f"got {ops_after}. This indicates PassResult handling is broken.",
        )

        # Verify graph is still valid
        device_edge_program.graph_module.graph.lint()

    def test_aoti_backend_pass_application_with_triton(self):
        """
        Regression test for AotiBackend.preprocess pass application with Triton.

        This test verifies that when custom passes (like ReplaceEdgeOpWithTritonOpPass)
        introduce operators that the default verifier doesn't recognize (like triton.sdpa),
        the pass application loop doesn't fail validation.

        The fix was to pass override_verifiers=[] to _update_exported_program_graph_module
        to skip validation during pass application.

        This test directly verifies that:
        1. CudaBackend.get_custom_passes() includes ReplaceEdgeOpWithTritonOpPass
        2. The pass application loop with override_verifiers=[] works correctly
        3. CSE pass returns PassResult with modified=True and new graph_module
        """
        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.exir.backend.compile_spec_schema import CompileSpec

        # Use a simple model (no SDPA) to test the pass application logic
        # The key thing we're testing is that override_verifiers=[] works
        model = self._build_qkv_model(hidden_dim=2048, block_size=128)
        example_input = (torch.randn(4, 128, 2048, device="cuda", dtype=torch.bfloat16),)
        exported = export(model, example_input, strict=False)

        # Get custom passes from CudaBackend WITH Triton enabled (default)
        compile_specs = []  # Default: triton_kernel_mode="ON"
        custom_passes = CudaBackend.get_custom_passes(compile_specs)

        # Verify we get all three passes including Triton
        pass_types = [type(p).__name__ for p in custom_passes]
        self.assertIn("CSEPass", pass_types)
        self.assertIn("FuseInt4WeightOnlyQuantMatmulPass", pass_types)
        self.assertIn("ReplaceEdgeOpWithTritonOpPass", pass_types)

        # Track that at least one pass returns PassResult with modified=True
        # This is the scenario that triggered the original bug
        modified_count = 0
        device_edge_program = exported

        for custom_pass in custom_passes:
            result = custom_pass(device_edge_program.graph_module)
            if isinstance(result, PassResult) and result.modified:
                modified_count += 1
                # This is the critical fix: override_verifiers=[] prevents
                # SpecViolationError when graph contains unknown operators
                device_edge_program = _update_exported_program_graph_module(
                    device_edge_program, result.graph_module, override_verifiers=[]
                )

        # Verify at least one pass modified the graph
        self.assertGreater(
            modified_count,
            0,
            "Expected at least one pass to return PassResult with modified=True",
        )

        # Verify graph is still valid (lint check)
        device_edge_program.graph_module.graph.lint()

        # Verify fusion occurred (the passes actually did something)
        ops_after = self._count_int4mm_ops(device_edge_program.graph_module)
        self.assertEqual(ops_after, 1, "Expected fusion to occur (3→1)")

    def test_fusion_numerical_correctness_with_edge_dialect(self):
        """
        Regression test for topological ordering bug in fusion pass.

        This test verifies that fusion produces numerically correct results when
        applied to a real edge-exported graph (not a manually constructed mock graph).

        The bug was that the fusion pass was inserting fused nodes (cat, fused_mm)
        at the wrong position in the graph - after placeholder nodes instead of
        after the input preprocessing nodes. This caused the fused_mm to reference
        inputs that hadn't been computed yet, breaking the computation.

        The fix was to use `inserting_before(first_int4mm)` instead of
        `inserting_after(max(weights + scales))` to maintain correct topological order.

        This test catches the bug by:
        1. Using a real edge-exported graph with proper preprocessing (constant_pad_nd, etc.)
        2. Properly updating the ExportedProgram after each pass (like aoti_backend.py does)
        3. Comparing eager output vs fused output numerically
        """
        # Skip if torchao not available
        try:
            from torchao.quantization import Int4WeightOnlyConfig, quantize_
        except ImportError:
            self.skipTest("torchao not available")

        # Skip if no CUDA or SM80+
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            self.skipTest("Requires SM80+ (A100 or newer)")

        from executorch.exir import EdgeCompileConfig, to_edge

        hidden_size = 256
        group_size = 128

        class SimpleQKVModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
                self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
                self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

            def forward(self, x):
                return self.q_proj(x) + self.k_proj(x) + self.v_proj(x)

        # Create and quantize model
        torch.manual_seed(42)
        module = SimpleQKVModule().to(dtype=torch.bfloat16, device="cuda").eval()

        int4_config = Int4WeightOnlyConfig(
            group_size=group_size,
            int4_packing_format="tile_packed_to_4d",
        )
        quantize_(module, int4_config)

        # Create input
        x = torch.randn(1, 16, hidden_size, dtype=torch.bfloat16, device="cuda")

        # Get eager output (ground truth)
        with torch.no_grad():
            eager_output = module(x)

        # Export to edge dialect (this creates the real graph structure with
        # preprocessing nodes like constant_pad_nd that exposed the bug)
        exported_program = export(module, (x,), strict=True)
        edge_program = to_edge(
            exported_program,
            compile_config=EdgeCompileConfig(_check_ir_validity=False)
        )

        # Get the exported program and apply passes using _update_exported_program_graph_module
        # This mirrors what aoti_backend.py does - passes return new graph_modules that must
        # be properly integrated back into the ExportedProgram
        ep = edge_program.exported_program()

        # Apply CSE pass and update the ExportedProgram
        cse_result = CSEPass()(ep.graph_module)
        if isinstance(cse_result, PassResult) and cse_result.modified:
            ep = _update_exported_program_graph_module(
                ep, cse_result.graph_module, override_verifiers=[]
            )

        # Count ops before fusion
        ops_before = self._count_int4mm_ops(ep.graph_module)
        self.assertEqual(ops_before, 3, "Expected 3 int4mm ops before fusion")

        # Apply fusion pass and update the ExportedProgram
        fusion_result = FuseInt4WeightOnlyQuantMatmulPass()(ep.graph_module)
        self.assertTrue(
            isinstance(fusion_result, PassResult) and fusion_result.modified,
            "Fusion pass should modify the graph"
        )
        ep = _update_exported_program_graph_module(
            ep, fusion_result.graph_module, override_verifiers=[]
        )

        # Count ops after fusion
        ops_after = self._count_int4mm_ops(ep.graph_module)
        self.assertEqual(ops_after, 1, "Expected 1 int4mm op after fusion")

        # Verify graph is topologically valid (this would have caught the bug)
        ep.graph_module.graph.lint()

        # Run fused graph and compare output
        # Now ep.module() returns the FUSED model because we properly updated ep
        with torch.no_grad():
            fused_output = ep.module()(x)

        # Verify numerical correctness
        diff = (eager_output - fused_output).abs()
        max_diff = diff.max().item()

        self.assertLess(
            max_diff,
            1e-3,
            f"Fused output differs from eager output by {max_diff}. "
            f"This indicates incorrect graph rewiring during fusion."
        )

    def test_fusion_split_outputs_contiguous_encoder_pattern(self):
        """
        Regression test for non-contiguous tensor_split outputs in encoder pattern.

        BUG: The fusion pass uses tensor_split to divide the fused output back into
        Q/K/V tensors. tensor_split creates non-contiguous views with incorrect strides:
        - Expected strides for [batch, seq, hidden]: [seq*hidden, hidden, 1]
        - Actual strides after split: [seq*3*hidden, 3*hidden, 1]

        This causes issues for encoder patterns with seq_len > 1 because:
        - Kernels assuming contiguous layout will read wrong memory locations
        - The stride[1] mismatch (3*hidden vs hidden) causes incorrect indexing

        The decoder (seq_len=1) is unaffected because dim 1 has size 1, making
        the stride irrelevant. This explains why encoder fails but decoder works.

        NOTE: In eager execution, the outputs may appear contiguous because
        subsequent view/reshape operations create contiguous copies. However,
        the FakeTensor metadata (used during AOTI compilation) correctly shows
        non-contiguous strides. This test checks the FakeTensor metadata to catch
        the bug that manifests during AOTI compilation.

        This test SHOULD FAIL until the fix is applied (adding .contiguous() after split).

        See: https://github.com/pytorch/executorch/issues/XXXXX
        """
        # Skip if torchao not available
        try:
            from torchao.quantization import Int4WeightOnlyConfig, quantize_
        except ImportError:
            self.skipTest("torchao not available")

        # Skip if no CUDA or SM80+
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            self.skipTest("Requires SM80+ (A100 or newer)")

        from executorch.exir import EdgeCompileConfig, to_edge

        hidden_size = 256
        group_size = 128
        # Use encoder-like seq_len (> 1) to trigger the bug
        seq_len = 64  # Encoder processes full sequence

        class QKVModule(torch.nn.Module):
            """Model that returns Q/K/V separately to check their contiguity."""
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
                self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
                self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

            def forward(self, x):
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)
                return q, k, v

        # Create and quantize model
        torch.manual_seed(42)
        module = QKVModule().to(dtype=torch.bfloat16, device="cuda").eval()

        int4_config = Int4WeightOnlyConfig(
            group_size=group_size,
            int4_packing_format="tile_packed_to_4d",
        )
        quantize_(module, int4_config)

        # Create encoder-like input (seq_len > 1)
        x = torch.randn(1, seq_len, hidden_size, dtype=torch.bfloat16, device="cuda")

        # Export to edge dialect
        exported_program = export(module, (x,), strict=True)
        edge_program = to_edge(
            exported_program,
            compile_config=EdgeCompileConfig(_check_ir_validity=False)
        )

        ep = edge_program.exported_program()

        # Apply CSE pass
        cse_result = CSEPass()(ep.graph_module)
        if isinstance(cse_result, PassResult) and cse_result.modified:
            ep = _update_exported_program_graph_module(
                ep, cse_result.graph_module, override_verifiers=[]
            )

        # Verify 3 int4mm ops before fusion
        ops_before = self._count_int4mm_ops(ep.graph_module)
        self.assertEqual(ops_before, 3, "Expected 3 int4mm ops before fusion")

        # Apply fusion pass
        fusion_result = FuseInt4WeightOnlyQuantMatmulPass()(ep.graph_module)
        self.assertTrue(
            isinstance(fusion_result, PassResult) and fusion_result.modified,
            "Fusion pass should modify the graph"
        )
        ep = _update_exported_program_graph_module(
            ep, fusion_result.graph_module, override_verifiers=[]
        )

        # Verify fusion occurred (3 -> 1)
        ops_after = self._count_int4mm_ops(ep.graph_module)
        self.assertEqual(ops_after, 1, "Expected 1 int4mm op after fusion")

        # Check FakeTensor metadata for the replacement nodes (contiguous nodes after getitem)
        # After the fix, the fusion pass adds .contiguous() after each getitem,
        # so we should check the contiguous nodes for proper metadata.
        contiguous_metadata = []
        for node in ep.graph_module.graph.nodes:
            if node.op == "call_function" and "contiguous" in str(node.target):
                if "val" in node.meta:
                    val = node.meta["val"]
                    if isinstance(val, torch.Tensor):
                        contiguous_metadata.append({
                            "name": node.name,
                            "shape": tuple(val.shape),
                            "stride": tuple(val.stride()),
                            "is_contiguous": val.is_contiguous(),
                        })

        # After the fix, there should be contiguous nodes with proper metadata
        self.assertGreater(len(contiguous_metadata), 0, "Expected contiguous nodes after fusion (fix applied)")

        for meta in contiguous_metadata:
            # Check that FakeTensor metadata shows contiguous tensors
            self.assertTrue(
                meta["is_contiguous"],
                f"FakeTensor for {meta['name']} should be contiguous.\n"
                f"Shape: {meta['shape']}, Strides: {meta['stride']}"
            )

    def test_fusion_split_outputs_decoder_pattern_contiguous(self):
        """
        Verify decoder pattern (seq_len=1) produces "contiguous" tensors in FakeTensor metadata.

        This test documents why the decoder works while encoder fails:
        - For seq_len=1, PyTorch considers the tensor contiguous even with
          stride[1] = 3*hidden because dim 1 has size 1 (stride is irrelevant).
        - For seq_len > 1 (encoder), the incorrect stride causes is_contiguous=False.

        This test SHOULD PASS (demonstrating the asymmetry between encoder/decoder).
        """
        # Skip if torchao not available
        try:
            from torchao.quantization import Int4WeightOnlyConfig, quantize_
        except ImportError:
            self.skipTest("torchao not available")

        # Skip if no CUDA or SM80+
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            self.skipTest("Requires SM80+ (A100 or newer)")

        from executorch.exir import EdgeCompileConfig, to_edge

        hidden_size = 256
        group_size = 128
        # Use decoder-like seq_len (= 1)
        seq_len = 1

        class QKVModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
                self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
                self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

            def forward(self, x):
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)
                return q, k, v

        # Create and quantize model
        torch.manual_seed(42)
        module = QKVModule().to(dtype=torch.bfloat16, device="cuda").eval()

        int4_config = Int4WeightOnlyConfig(
            group_size=group_size,
            int4_packing_format="tile_packed_to_4d",
        )
        quantize_(module, int4_config)

        # Create decoder-like input (seq_len = 1)
        x = torch.randn(1, seq_len, hidden_size, dtype=torch.bfloat16, device="cuda")

        # Export to edge dialect
        exported_program = export(module, (x,), strict=True)
        edge_program = to_edge(
            exported_program,
            compile_config=EdgeCompileConfig(_check_ir_validity=False)
        )

        ep = edge_program.exported_program()

        # Apply CSE + Fusion passes
        cse_result = CSEPass()(ep.graph_module)
        if isinstance(cse_result, PassResult) and cse_result.modified:
            ep = _update_exported_program_graph_module(
                ep, cse_result.graph_module, override_verifiers=[]
            )

        fusion_result = FuseInt4WeightOnlyQuantMatmulPass()(ep.graph_module)
        if isinstance(fusion_result, PassResult) and fusion_result.modified:
            ep = _update_exported_program_graph_module(
                ep, fusion_result.graph_module, override_verifiers=[]
            )

        # Verify fusion occurred
        ops_after = self._count_int4mm_ops(ep.graph_module)
        self.assertEqual(ops_after, 1, "Expected 1 int4mm op after fusion")

        # Check FakeTensor metadata for contiguous nodes (after the fix is applied)
        contiguous_metadata = []
        for node in ep.graph_module.graph.nodes:
            if node.op == "call_function" and "contiguous" in str(node.target):
                if "val" in node.meta:
                    val = node.meta["val"]
                    if isinstance(val, torch.Tensor):
                        contiguous_metadata.append({
                            "name": node.name,
                            "shape": tuple(val.shape),
                            "stride": tuple(val.stride()),
                            "is_contiguous": val.is_contiguous(),
                        })

        # After the fix, there should be contiguous nodes
        self.assertGreater(len(contiguous_metadata), 0, "Expected contiguous nodes after fusion")

        # For seq_len=1, FakeTensor should show contiguous=True
        for meta in contiguous_metadata:
            self.assertTrue(
                meta["is_contiguous"],
                f"Decoder FakeTensor for {meta['name']} (seq_len=1) should be contiguous.\n"
                f"Shape: {meta['shape']}, Strides: {meta['stride']}"
            )


if __name__ == "__main__":
    unittest.main()
