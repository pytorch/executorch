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


def _ops_outside_delegates(program):
    return [
        node.target.__name__
        for node in program.exported_program().graph.nodes
        if node.op == "call_function" and hasattr(node.target, "__name__")
    ]


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

    def test_enable_bf16_leaves_ops_xnnpack_cannot_define_in_bf16(self):
        """
        XNNPACK refuses these ops on bf16 tensors when the program loads, so even
        with enable_bf16 their bf16 nodes must run on the portable kernels. The
        same modules are delegated in fp32, so bf16 is what keeps them out.
        """

        class Mean(torch.nn.Module):
            def forward(self, x):
                return x.mean((-1, -2), keepdim=True)

        class Softmax(torch.nn.Module):
            def forward(self, x):
                return torch.softmax(x, -1)

        def lower(module, x):
            return to_edge_transform_and_lower(
                export(module, (x,)),
                partitioner=[XnnpackPartitioner(enable_bf16=True)],
            )

        modules = {
            "avg_pool2d": torch.nn.AvgPool2d(2),
            "batch_norm": torch.nn.BatchNorm2d(4),
            "conv_batch_norm": torch.nn.Sequential(
                torch.nn.Conv2d(4, 4, 3), torch.nn.BatchNorm2d(4)
            ),
            "max_pool2d": torch.nn.MaxPool2d(2),
            "mean": Mean(),
            "softmax": Softmax(),
            "upsample_bilinear2d": torch.nn.Upsample(scale_factor=2, mode="bilinear"),
        }
        x = torch.randn(1, 4, 8, 8)
        for name, module in modules.items():
            with self.subTest(name):
                module.eval()
                self.assertIn(
                    "executorch_call_delegate", _ops_outside_delegates(lower(module, x))
                )
                module, bf16_x = module.to(torch.bfloat16), x.to(torch.bfloat16)
                program = lower(module, bf16_x)
                self.assertNotIn(
                    "executorch_call_delegate", _ops_outside_delegates(program)
                )
                output = _load_for_executorch_from_buffer(
                    program.to_executorch().buffer
                ).forward((bf16_x,))[0]
                torch.testing.assert_close(output, module(bf16_x))

    def test_enable_bf16_fuses_batch_norm_into_a_linear(self):
        """
        XNNPACK cannot define a bf16 BatchNorm on its own, but one that follows a
        linear folds into it, so the whole bf16 block still runs in XNNPACK. It
        does not fold when the linear output has another user, so then it stays
        on the portable kernels.
        """

        class LinearBatchNorm(torch.nn.Module):
            def __init__(self, reuse_linear_output):
                super().__init__()
                self.linear = torch.nn.Linear(8, 4)
                self.batch_norm = torch.nn.BatchNorm1d(4)
                self.batch_norm.running_mean.uniform_(-1, 1)
                self.batch_norm.running_var.uniform_(0.5, 2)
                self.reuse_linear_output = reuse_linear_output

            def forward(self, x):
                y = self.linear(x)
                z = self.batch_norm(y)
                return z + y if self.reuse_linear_output else z

        x = torch.randn(2, 8, dtype=torch.bfloat16)
        delegated = {"executorch_call_delegate", "getitem"}
        for reuse_linear_output, ops_outside in (
            (False, delegated),
            (True, delegated | {"aten._native_batch_norm_legit_no_training.default"}),
        ):
            with self.subTest(reuse_linear_output=reuse_linear_output):
                module = LinearBatchNorm(reuse_linear_output).eval().to(torch.bfloat16)
                program = to_edge_transform_and_lower(
                    export(module, (x,)),
                    partitioner=[XnnpackPartitioner(enable_bf16=True)],
                )
                self.assertEqual(set(_ops_outside_delegates(program)), ops_outside)
                output = _load_for_executorch_from_buffer(
                    program.to_executorch().buffer
                ).forward((x,))[0]
                torch.testing.assert_close(output, module(x), atol=1e-2, rtol=1e-2)

    def test_enable_bf16_runs_an_attention_block(self):
        """
        The linear layers and both batch matrix multiplications of a bf16 attention
        block run in XNNPACK, and the program loads and matches eager bf16.
        """

        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q = torch.nn.Linear(16, 16)
                self.k = torch.nn.Linear(16, 16)
                self.v = torch.nn.Linear(16, 16)
                self.o = torch.nn.Linear(16, 16, bias=False)

            def forward(self, x):
                scores = self.q(x) @ self.k(x).transpose(1, 2) * 0.25
                return self.o(torch.softmax(scores, dim=-1) @ self.v(x))

        module = Attention().eval().to(torch.bfloat16)
        x = torch.randn(2, 8, 16, dtype=torch.bfloat16)
        program = to_edge_transform_and_lower(
            export(module, (x,)), partitioner=[XnnpackPartitioner(enable_bf16=True)]
        )
        ops = _ops_outside_delegates(program)
        # The expand copies that matmul adds split the block into 3 delegates.
        self.assertEqual(ops.count("executorch_call_delegate"), 3)
        self.assertIn("aten._softmax.default", ops)
        matmuls = {
            "aten.addmm.default",
            "aten.bmm.default",
            "aten.linear.default",
            "aten.mm.default",
        }
        self.assertEqual(set(ops) & matmuls, set())
        output = _load_for_executorch_from_buffer(
            program.to_executorch().buffer
        ).forward((x,))[0]
        torch.testing.assert_close(output, module(x), atol=1e-2, rtol=1e-2)
