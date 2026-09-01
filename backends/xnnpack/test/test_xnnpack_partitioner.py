# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import logging
import unittest

import torch
import torch.nn.functional as F

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)
from torch.export import export


class TestXnnpackPartitioner(unittest.TestCase):
    """Test cases for XnnpackPartitioner functionality and deprecation warnings."""

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    def test_deprecation_warning_for_to_backend_workflow(self):
        """
        Test that the deprecated to_edge + to_backend workflow shows a deprecation warning.
        """
        model = self.SimpleModel()
        x = torch.randn(1, 10)

        exported_model = export(model, (x,))

        # Capture log output to check for deprecation warning
        log_capture_string = io.StringIO()
        ch = logging.StreamHandler(log_capture_string)
        ch.setLevel(logging.WARNING)

        logger = logging.getLogger(
            "executorch.backends.xnnpack.partition.xnnpack_partitioner"
        )
        logger.addHandler(ch)
        logger.setLevel(logging.WARNING)

        edge = to_edge(exported_model)
        partitioner = XnnpackPartitioner()

        edge.to_backend(partitioner)

        log_contents = log_capture_string.getvalue()
        self.assertIn("DEPRECATION WARNING", log_contents)
        self.assertIn("to_edge() + to_backend()", log_contents)
        self.assertIn("to_edge_transform_and_lower()", log_contents)

    def test_no_warning_for_to_edge_transform_and_lower_workflow(self):
        """
        Test that the recommended to_edge_transform_and_lower workflow does NOT show a deprecation warning.
        """

        model = self.SimpleModel()
        x = torch.randn(1, 10)

        exported_model = export(model, (x,))

        # Capture log output to check for deprecation warning
        log_capture_string = io.StringIO()
        ch = logging.StreamHandler(log_capture_string)
        ch.setLevel(logging.WARNING)

        logger = logging.getLogger(
            "executorch.backends.xnnpack.partition.xnnpack_partitioner"
        )
        logger.addHandler(ch)
        logger.setLevel(logging.WARNING)

        partitioner = XnnpackPartitioner()

        to_edge_transform_and_lower(exported_model, partitioner=[partitioner])

        log_contents = log_capture_string.getvalue()
        self.assertNotIn("DEPRECATION WARNING", log_contents)

    def test_multi_method_partitioning_with_shared_weights(self):
        """
        Test that multi-method models with shared weights are correctly partitioned.
        Verify that:
        1. Both methods are fully lowered to XNNPACK.
        2. Constants are not duplicated between named data and constant buffers.
        3. Program executes correctly.
        """

        class MultiMethodModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(8, 16)
                self.linear2 = torch.nn.Linear(16, 8)

            def forward(self, x):
                return self.linear2(F.sigmoid(self.linear(x)))

            def forward_2(self, x):
                return self.linear2(F.relu(self.linear(x)))

            def example_inputs(self):
                return (torch.randn(1, 8),)

        model = MultiMethodModel()

        # Get eager reference output.
        example_inputs = model.example_inputs()
        with torch.no_grad():
            fwd1_eager = model.forward(*example_inputs)
            fwd2_eager = model.forward_2(*example_inputs)

        # Export both methods
        ep_fwd = export(model, model.example_inputs(), strict=True)
        # Patch the forward, as export only traces the 'forward' method.
        model.forward = model.forward_2
        ep_fwd_2 = export(model, model.example_inputs(), strict=True)

        # Convert to edge and lower to executorch
        edge = to_edge({"forward": ep_fwd, "forward_2": ep_fwd_2})
        lowered = edge.to_backend(XnnpackPartitioner(force_fp32_dynamic_linear=True))
        executorch = lowered.to_executorch()

        # Check that graph is fully delegated.
        nodes_1 = list(lowered._edge_programs["forward"].graph.nodes)
        nodes_2 = list(lowered._edge_programs["forward_2"].graph.nodes)
        self.assertEqual(len(nodes_1), 5)
        self.assertEqual(len(nodes_2), 5)
        expected_node_names = [
            "x",
            "lowered_module_0",
            "executorch_call_delegate",
            "getitem",
            "output_1",
        ]
        for n in expected_node_names:
            self.assertTrue(any(node.name == n for node in nodes_1))
            self.assertTrue(any(node.name == n for node in nodes_2))

        # Check that weights are not duplicated.
        self.assertEqual(len(executorch._named_data.pte_data), 4)
        self.assertEqual(len(executorch._named_data.buffers), 4)
        self.assertEqual(len(executorch._named_data.external_data), 0)

        # Check that there are no constant buffers (besides the placeholder).
        self.assertEqual(len(executorch._emitter_output.program.constant_buffer), 1)

        # Check for model correctness.
        executorch_module = _load_for_executorch_from_buffer(executorch.buffer)
        fwd1_et = executorch_module.run_method("forward", example_inputs)
        fwd2_et = executorch_module.run_method("forward_2", example_inputs)
        self.assertTrue(torch.allclose(fwd1_eager, fwd1_et[0], 1e-3))
        self.assertTrue(torch.allclose(fwd2_eager, fwd2_et[0], 1e-3))

    def test_parametrized_weight_is_folded_before_partitioning(self):
        """
        A weight computed from parameters (here weight_norm) is folded into a
        constant before partitioning, so the convolution is delegated instead
        of falling back to the portable kernels with the weight computation.
        """

        class ParametrizedConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.utils.parametrizations.weight_norm(
                    torch.nn.Conv1d(4, 4, 3)
                )

            def forward(self, x):
                return self.conv(x)

        model = ParametrizedConv().eval()
        example_inputs = (torch.randn(1, 4, 8),)
        eager = model(*example_inputs)

        edge = to_edge_transform_and_lower(
            export(model, example_inputs), partitioner=[XnnpackPartitioner()]
        )
        call_functions = [
            node
            for node in edge.exported_program().graph_module.graph.nodes
            if node.op == "call_function"
        ]
        delegates = [
            node
            for node in call_functions
            if node.target == torch.ops.higher_order.executorch_call_delegate
        ]
        self.assertEqual(len(delegates), 1)
        # The delegate call and the getitem on its output are all that is left.
        self.assertEqual(len(call_functions), 2)

        executorch_module = _load_for_executorch_from_buffer(
            edge.to_executorch().buffer
        )
        self.assertTrue(
            torch.allclose(executorch_module.forward(example_inputs)[0], eager, 1e-5)
        )

    def test_pre_decomposition_folding_keeps_quantization_primitives(self):
        """
        Folding must not touch the Q/DQ chain that convert_pt2e leaves on a
        quantized weight, or the weight would be dequantized at export time.
        """
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            get_symmetric_quantization_config,
            XNNPACKQuantizer,
        )
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

        model = self.SimpleModel().eval()
        example_inputs = (torch.randn(2, 10),)
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
        prepared = prepare_pt2e(export(model, example_inputs).module(), quantizer)
        prepared(*example_inputs)
        converted = convert_pt2e(prepared)

        def quant_targets(ep):
            return sorted(
                str(node.target)
                for node in ep.graph.nodes
                if node.op == "call_function"
                and "quantized_decomposed" in str(node.target)
            )

        class GroupwiseLinear(torch.nn.Module):
            """A weight stored as int8 groups, the way 4-bit LLM exports do."""

            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "weight", torch.randint(-8, 8, (8, 16), dtype=torch.int8)
                )
                self.register_buffer("scales", torch.rand(8, 2))
                self.register_buffer("zeros", torch.zeros(8, 2, dtype=torch.int8))

            def forward(self, x):
                weight = torch.ops.quantized_decomposed.dequantize_per_channel_group(
                    self.weight, self.scales, self.zeros, -8, 7, torch.int8, 8, x.dtype
                )
                return torch.nn.functional.linear(x, weight)

        for model, example_inputs in (
            (converted, example_inputs),
            (GroupwiseLinear(), (torch.randn(2, 16),)),
        ):
            exported = export(model, example_inputs)
            before = quant_targets(exported)
            self.assertGreater(len(before), 0)
            after = quant_targets(
                XnnpackPartitioner().transform_for_pre_decomposition(exported)
            )
            self.assertEqual(before, after)
