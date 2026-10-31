# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)
from executorch.backends.xnnpack.test.tester import (
    Quantize,
    Tester,
    ToEdgeTransformAndLower,
)

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

    def test_fp32_permute_max_rank(self):
        # XNNPACK allows at most 6 dims, so a 6-D permute is delegated and a 7-D
        # one is not.
        delegate = "torch.ops.higher_order.executorch_call_delegate"
        permute = "executorch_exir_dialects_edge__ops_aten_permute_copy_default"
        for rank, delegates in ((6, 1), (7, 0)):
            dims = list(reversed(range(rank)))
            (
                Tester(self.Permute(dims), (torch.randn([2] * rank),))
                .export()
                .to_edge_transform_and_lower()
                .check_count({delegate: delegates, permute: 1 - delegates})
                .to_executorch()
                .serialize()
                .run_method_and_compare_outputs()
            )

    def _test_permute_above_max_rank(self, quant_config=None, partitioner=None):
        # A delegate that holds a tensor above 6 dims fails to load, so the 8-D
        # views and permute stay outside and the mul is delegated. The linear's
        # 2-D matmul is delegated too, unless its activation is quantized in 8-D.
        tester = Tester(self.LinearPermute8d().eval(), (torch.randn(2, 64, 4),))
        if quant_config is not None:
            tester.quantize(Quantize(quantization_config=quant_config))
        tester.export().to_edge_transform_and_lower(
            ToEdgeTransformAndLower([partitioner]) if partitioner else None
        )
        addmm = ["executorch_exir_dialects_edge__ops_aten_addmm_default"]
        if quant_config is None:
            tester.check_not(addmm)
        else:
            tester.check(addmm)
        (
            tester.check_count(
                {
                    "torch.ops.higher_order.executorch_call_delegate": 2,
                    "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
                }
            )
            .check_not(["executorch_exir_dialects_edge__ops_aten_mul_Tensor"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(qtol=1)
        )

    def test_fp32_permute_above_max_rank(self):
        for force_non_static_weights in (False, True):
            self._test_permute_above_max_rank(
                partitioner=XnnpackPartitioner(
                    force_non_static_weights_for_f32_linear=force_non_static_weights
                )
            )

    def test_qs8_permute_above_max_rank(self):
        self._test_permute_above_max_rank(get_symmetric_quantization_config())

    def test_qd8_permute_above_max_rank(self):
        self._test_permute_above_max_rank(
            get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
        )

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
