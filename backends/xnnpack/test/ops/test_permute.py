# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester
from executorch.exir.lowered_backend_module import get_lowered_backend_modules

# Buck preloads the quantized kernels instead of shipping this module.
try:
    import executorch.extension.pybindings.portable_lib  # noqa[F401]
    import executorch.kernels.quantized  # noqa[F401]
except ImportError:
    pass


class TestPermute(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    class Permute(torch.nn.Module):
        def __init__(self, dims):
            self.dims = dims
            super().__init__()

        def forward(self, x):
            y = x + x
            z = torch.permute(y, self.dims)
            return z

    class PermuteCopy(torch.nn.Module):
        def __init__(self, dims):
            self.dims = dims
            super().__init__()

        def forward(self, x):
            y = x + x
            z = torch.permute_copy(y, self.dims)
            return z

    def _test_permute(self, inputs):
        (
            Tester(self.Permute([0, 2, 3, 1]), inputs)
            .export()
            .check_count({"torch.ops.aten.permute.default": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_permute_copy_default"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp16_permute(self):
        inputs = (torch.randn(1, 1, 4, 4).to(torch.float16),)
        self._test_permute(inputs)

    def test_fp32_permute(self):
        inputs = (torch.randn(1, 1, 4, 4),)
        self._test_permute(inputs)

    def test_fp32_permute_negative_dim(self):
        inputs = (torch.randn(1, 1, 4, 4),)
        (
            Tester(self.Permute([0, -2, -1, 1]), inputs)
            .export()
            .check_count({"torch.ops.aten.permute.default": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_permute_copy_default"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_permute_copy(self):
        inputs = (torch.randn(1, 1, 4, 4),)
        (
            Tester(self.PermuteCopy([0, 2, 3, 1]), inputs)
            .export()
            .check_count({"torch.ops.aten.permute_copy.default": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_permute_copy_default"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    class LinearPermute8d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, x):
            y = self.linear(x.reshape(2, 2, 2, 2, 2, 2, 2, 4))
            return y.permute(0, 2, 5, 3, 6, 1, 4, 7).reshape(2, 64, -1) * 2

    def _test_permute_above_max_rank(self, quantize):
        # XNNPACK allows at most 6 dims, and a delegate that holds a bigger tensor
        # fails to load. The linear also pulls in its 8-D views as dependencies.
        tester = Tester(self.LinearPermute8d().eval(), (torch.randn(2, 64, 4),))
        if quantize:
            tester.quantize()
        tester.export().to_edge_transform_and_lower()

        lowered_modules = get_lowered_backend_modules(
            tester.get_artifact().exported_program().graph_module
        )
        self.assertGreater(len(lowered_modules), 0)
        for lowered_module in lowered_modules:
            for node in lowered_module.original_module.graph.nodes:
                val = node.meta.get("val")
                if isinstance(val, torch.Tensor):
                    self.assertLessEqual(val.dim(), 6, node.format_node())

        tester.to_executorch().serialize().run_method_and_compare_outputs(qtol=1)

    def test_fp32_permute_above_max_rank(self):
        self._test_permute_above_max_rank(quantize=False)

    def test_qs8_permute_above_max_rank(self):
        self._test_permute_above_max_rank(quantize=True)

    def test_qs8_permute(self):
        inputs = (torch.randn(1, 1, 4, 4),)
        (
            Tester(self.Permute([0, 2, 3, 1]), inputs)
            .quantize()
            .export()
            .check_node_count(
                {
                    torch.ops.aten.permute.default: 1,
                    torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
                }
            )
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_permute_copy_default",
                    "torch.ops.quantized_decomposed",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_qs8_permute_copy(self):
        inputs = (torch.randn(1, 1, 4, 4),)
        (
            Tester(self.PermuteCopy([0, 2, 3, 1]), inputs)
            .quantize()
            .export()
            .check_node_count(
                {
                    torch.ops.aten.permute_copy.default: 1,
                    torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
                }
            )
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_permute_copy_default",
                    "torch.ops.quantized_decomposed",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
