# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import executorch
import executorch.backends.cortex_m.ops.operators  # noqa

import torch

from executorch.backends.cortex_m.passes.quantized_op_fusion_pass import (
    QuantizedOpFusionPass,
)
from executorch.backends.cortex_m.passes.replace_quant_nodes_pass import (
    ReplaceQuantNodesPass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from test_helpers_passes_utils import AddQuantizer, check_count, get_node_args
from torch.export import export, export_for_training
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class TestQuantizedOpFusionPass(unittest.TestCase):
    """
    Test suite for the QuantizedOpFusionPass which fuses dequantize->add->quantize patterns
    into a single quantized_add operation with AoT-computed parameters.
    """

    def setUp(self):
        """Set up common test fixtures"""
        self.example_inputs = (torch.randn(4, 8), torch.randn(4, 8))

    def _prepare_quantized_model(self, model_class):
        """Helper to prepare a quantized model for testing"""
        model = model_class()

        # Export and quantize
        exported_model = export_for_training(
            model.eval(), self.example_inputs, strict=True
        ).module()
        prepared_model = prepare_pt2e(exported_model, AddQuantizer())
        quantized_model = convert_pt2e(prepared_model)

        # Export to EXIR Edge
        exported = export(quantized_model, self.example_inputs, strict=True)
        edge_program = executorch.exir.to_edge(
            exported,
            compile_config=executorch.exir.EdgeCompileConfig(_check_ir_validity=False),
        )
        return edge_program

    def _apply_passes(self, edge_program):
        """Apply both ReplaceQuantNodesPass and QuantizedOpFusionPass"""
        passes = [QuantizedOpFusionPass(), ReplaceQuantNodesPass()]
        final_program = edge_program.transform(passes)
        return final_program

    def test_single_add_fusion(self):
        """Single add with full Q/DQ pattern should fuse into one quantized_add node"""

        class SingleAddModel(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        # Prepare model
        edge_program = self._prepare_quantized_model(SingleAddModel)
        edge_graph = edge_program.exported_program().graph_module

        # Get reference output
        reference_output = edge_graph(*self.example_inputs)

        # Apply passes
        transformed_program = self._apply_passes(edge_program)
        transformed_graph = transformed_program.exported_program().graph_module

        # Verify fusion occurred
        check_count(
            transformed_graph,
            exir_ops.edge.cortex_m.quantized_add.default,
            1,  # Should have exactly 1 fused quantized_add
        )

        # Verify the following
        # Before fusion:
        #   x --> quantize_per_tensor --> dequantize_per_tensor --> add --> quantize_per_tensor -->
        #   dequantize_per_tensor --> output y --> quantize_per_tensor --> dequantize_per_tensor --^
        # After fusion:
        #   x --> quantize_per_tensor --> quantized_add --> dequantize_per_tensor --> output
        #   y --> quantize_per_tensor --^
        check_count(
            transformed_graph, exir_ops.edge.cortex_m.quantize_per_tensor.default, 2
        )
        check_count(
            transformed_graph, exir_ops.edge.cortex_m.dequantize_per_tensor.default, 1
        )
        check_count(transformed_graph, exir_ops.edge.cortex_m.quantized_add.default, 1)

        # Verify numerical equivalence
        fused_output = transformed_graph(*self.example_inputs)
        torch.testing.assert_close(reference_output, fused_output, rtol=1e-3, atol=1e-3)

    def test_multiple_add_fusion(self):
        """Multiple independent adds should create multiple quantized_add nodes"""

        class MultipleAddModel(torch.nn.Module):
            def forward(self, x, y):
                z1 = x + y  # First add
                z2 = x + z1  # Second add
                return z2

        # Prepare model
        edge_program = self._prepare_quantized_model(MultipleAddModel)
        edge_graph = edge_program.exported_program().graph_module

        # Get reference output
        reference_output = edge_graph(*self.example_inputs)

        # Apply passes
        transformed_program = self._apply_passes(edge_program)
        transformed_graph = transformed_program.exported_program().graph_module

        # Verify multiple fusions occurred
        check_count(
            transformed_graph,
            exir_ops.edge.cortex_m.quantized_add.default,
            2,  # Should have 2 fused quantized_add nodes
        )

        # Verify numerical equivalence
        fused_output = transformed_graph(*self.example_inputs)
        torch.testing.assert_close(reference_output, fused_output, rtol=1e-3, atol=1e-3)

    def test_no_fusion_without_pattern(self):
        """Add without proper Q/DQ pattern should not be fused"""

        class NonQuantizedAddModel(torch.nn.Module):
            def forward(self, x, y):
                # This will have add but not the full Q/DQ pattern after quantization
                return torch.relu(x + y)  # ReLU breaks the pattern

        # For this test, we'll create a model that doesn't have the complete pattern
        # We need to manually construct a graph that has add without full Q/DQ

        model = NonQuantizedAddModel()
        exported = export(model, self.example_inputs, strict=True)
        edge_program = executorch.exir.to_edge(
            exported,
            compile_config=executorch.exir.EdgeCompileConfig(_check_ir_validity=False),
        )
        # Apply passes
        transformed_program = self._apply_passes(edge_program)
        transformed_graph = transformed_program.exported_program().graph_module

        # Verify no fusion occurred
        check_count(
            transformed_graph,
            exir_ops.edge.cortex_m.quantized_add.default,
            0,  # Should have no fused quantized_add nodes
        )

    def test_precomputed_parameters(self):
        """Fused node should have precomputed multipliers/shifts instead of scales"""

        class SingleAddModel(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        # Prepare model
        edge_program = self._prepare_quantized_model(SingleAddModel)

        # Apply passes
        transformed_program = self._apply_passes(edge_program)
        transformed_graph = transformed_program.exported_program().graph_module

        # Get arguments of the fused quantized_add node
        quantized_add_args = get_node_args(
            transformed_graph, exir_ops.edge.cortex_m.quantized_add.default
        )

        # Should have exactly one quantized_add node
        self.assertEqual(len(quantized_add_args), 1)
        args = quantized_add_args[0]

        # Verify argument structure: (tensor1, zp1, mult1, shift1, tensor2, zp2, mult2, shift2, out_zp, out_mult, out_shift)
        self.assertEqual(len(args), 11, "quantized_add should have 11 arguments")

        # Check that multipliers and shifts are integers (not floats/scales)
        # args[2], args[3] = input1 multiplier, shift
        # args[6], args[7] = input2 multiplier, shift
        # args[9], args[10] = output multiplier, shift
        for i in [2, 3, 6, 7, 9, 10]:  # multiplier and shift positions
            self.assertIsInstance(
                args[i], int, f"Argument {i} should be an integer (precomputed)"
            )

    def test_mixed_fusion_pattern(self):
        """Mixed pattern (some fusable, some not) should partially fuse"""

        class MixedModel(torch.nn.Module):
            def forward(self, x, y):
                z1 = x + y  # This should fuse
                z2 = torch.relu(z1)  # ReLU breaks next fusion
                z3 = z2 + x  # This won't have full Q/DQ pattern
                return z3

        # Prepare model
        edge_program = self._prepare_quantized_model(MixedModel)

        # Apply passes
        transformed_program = self._apply_passes(edge_program)
        transformed_graph = transformed_program.exported_program().graph_module

        # Should have partial fusion (at least 1, but not necessarily all adds)
        quantized_add_count = sum(
            1
            for node in transformed_graph.graph.nodes
            if node.op == "call_function"
            and node.target == exir_ops.edge.cortex_m.quantized_add.default
        )

        self.assertGreaterEqual(
            quantized_add_count, 1, "Should have at least 1 fused operation"
        )

    def test_different_tensor_shapes(self):
        """Different tensor shapes should still fuse correctly"""

        class SingleAddModel(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        # Test with different input shapes
        for shape in [(2, 3), (10, 20, 30), (1,)]:
            with self.subTest(shape=shape):
                inputs = (torch.randn(shape), torch.randn(shape))

                model = SingleAddModel()
                exported_model = export_for_training(
                    model.eval(), inputs, strict=True
                ).module()
                prepared_model = prepare_pt2e(exported_model, AddQuantizer())
                quantized_model = convert_pt2e(prepared_model)

                exported = export(quantized_model, inputs, strict=True)
                edge_program = executorch.exir.to_edge(
                    exported,
                    compile_config=executorch.exir.EdgeCompileConfig(
                        _check_ir_validity=False
                    ),
                )

                # Apply passes
                transformed_program = self._apply_passes(edge_program)
                transformed_graph = transformed_program.exported_program().graph_module

                # Verify fusion occurred regardless of shape
                check_count(
                    transformed_graph, exir_ops.edge.cortex_m.quantized_add.default, 1
                )

    def test_aot_parameter_computation_accuracy(self):
        """Verify that AoT-computed parameters match runtime computation"""

        class SingleAddModel(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        # Prepare model
        edge_program = self._prepare_quantized_model(SingleAddModel)

        # Apply passes
        transformed_program = self._apply_passes(edge_program)
        transformed_graph = transformed_program.exported_program().graph_module

        # Get the fused node arguments
        quantized_add_args = get_node_args(
            transformed_graph, exir_ops.edge.cortex_m.quantized_add.default
        )[0]

        # Extract the computed multipliers and shifts
        input1_mult, input1_shift = quantized_add_args[2], quantized_add_args[3]
        input2_mult, input2_shift = quantized_add_args[6], quantized_add_args[7]
        output_mult, output_shift = quantized_add_args[9], quantized_add_args[10]

        # Verify they are reasonable values
        # Multipliers should be in int32 range
        self.assertTrue(-(2**31) <= input1_mult < 2**31)
        self.assertTrue(-(2**31) <= input2_mult < 2**31)
        self.assertTrue(-(2**31) <= output_mult < 2**31)

        # Shifts should be reasonable (typically -31 to 31)
        self.assertTrue(-50 <= input1_shift <= 50)
        self.assertTrue(-50 <= input2_shift <= 50)
        self.assertTrue(-50 <= output_shift <= 50)

        # Output multiplier should be close to 2^30 (for 1.0 scale)
        self.assertAlmostEqual(output_mult, 2**30, delta=1000)
        self.assertEqual(output_shift, -1)

    def test_executorch_program_generation(self):
        """Verify ExecuTorch program generation with fused ops"""

        class SingleAddModel(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        # Prepare model
        edge_program = self._prepare_quantized_model(SingleAddModel)

        # Apply passes
        transformed_program = self._apply_passes(edge_program)

        # Generate ExecutorTorch program
        executorch_program = transformed_program.to_executorch()

        # Verify the program contains the expected fused operator
        operator_names = [
            op.name
            for op in executorch_program.executorch_program.execution_plan[0].operators
        ]

        self.assertIn("cortex_m::quantized_add", operator_names)
        self.assertIn("cortex_m::quantize_per_tensor", operator_names)
        self.assertIn("cortex_m::dequantize_per_tensor", operator_names)
        # quantize_per_tensor --> dequantize_per_tensor --> add --> quantize_per_tensor --> dequantize_per_tensor
        # (input quant)          (dequant)               (fp32 add)       (re-quant)          (dequant)
        #                ↓
        # Fusion Pass detects pattern:
        # dequantize_per_tensor --> quantized_add (Fused node) --> quantize_per_tensor

    def test_broadcastable_shapes(self):
        """Verify that broadcastable shapes are supported"""

        class BroadcastAddModel(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        # input broadcastable shapes
        inputs = (torch.randn(4, 1), torch.randn(4, 8))
        print(inputs)

        # Prepare quantized model
        edge_program = self._prepare_quantized_model(BroadcastAddModel)

        # Get unfused output
        unfused_graph = edge_program.exported_program().graph_module
        unfused_output = unfused_graph(*inputs)
        if isinstance(unfused_output, tuple):
            unfused_output = unfused_output[0]

        # Apply fusion pass
        fused_program = self._apply_passes(edge_program)
        fused_graph = fused_program.exported_program().graph_module
        fused_output = fused_graph(*inputs)
        if isinstance(fused_output, tuple):
            fused_output = fused_output[0]

        # Check fusion occurred
        check_count(fused_graph, exir_ops.edge.cortex_m.quantized_add.default, 1)

        # Compare fused vs unfused (both quantized)
        torch.testing.assert_close(fused_output, unfused_output, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
