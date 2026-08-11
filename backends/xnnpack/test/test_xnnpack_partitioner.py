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
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from executorch.backends.xnnpack.utils.configs import get_transform_passes
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


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

    def test_quantized_view_with_mismatched_qparams_stays_portable(self):
        class ConvPoolFlattenLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 4, 3)
                self.linear = torch.nn.Linear(4, 2)

            def forward(self, x):
                x = self.conv(x)
                x = F.adaptive_avg_pool2d(x, (1, 1))
                return self.linear(torch.flatten(x, 1))

        inputs = (torch.randn(1, 3, 8, 8),)
        exported = export(ConvPoolFlattenLinear().eval(), inputs, strict=False)
        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=False)
        )
        prepared = prepare_pt2e(exported.module(), quantizer)
        prepared(*inputs)
        converted = convert_pt2e(prepared)

        output_quant = next(
            node
            for node in converted.graph.nodes
            if "quantize_per_tensor" in str(node.target)
            and any("flatten" in str(inp.target) for inp in node.all_input_nodes)
        )
        output_scale = output_quant.args[1] * 2
        output_quant.args = (
            output_quant.args[0],
            output_scale,
            *output_quant.args[2:],
        )
        output_dequant = next(iter(output_quant.users))
        output_dequant.args = (
            output_dequant.args[0],
            output_scale,
            *output_dequant.args[2:],
        )
        converted.recompile()

        lowered = to_edge_transform_and_lower(
            export(converted, inputs, strict=False),
            partitioner=[XnnpackPartitioner()],
            transform_passes=get_transform_passes(),
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False, _skip_dim_order=True
            ),
        )
        self.assertTrue(
            any(
                "view_copy" in str(node.target)
                for node in lowered.exported_program().graph.nodes
            )
        )

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
