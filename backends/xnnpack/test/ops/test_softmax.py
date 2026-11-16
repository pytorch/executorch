# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.test.tester import Tester
from executorch.backends.xnnpack.test.tester.tester import ToEdgeTransformAndLower


class TestSoftmax(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    class Softmax(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            return torch.nn.Softmax(dim=self.dim)(x)

    def _test_softmax(self, inputs, enable_bf16=False):
        # Dim can be either the last dimension index or -1 (last dimension),
        # as xnnpack only supports softmax on the last dimension.
        valid_dims = [len(inputs[0]) - 1, -1]

        dynamic_shape = {}
        for i in range(len(inputs[0].shape)):
            dynamic_shape[i] = torch.export.Dim(f"dynamic_dim{i}", min=1, max=100)
        dynamic_shape = (dynamic_shape,)

        # XNNPACK computes a bf16 softmax in fp32 and rounds back to bf16,
        # so it needs the looser bf16 tolerance.
        atol = rtol = 2e-2 if enable_bf16 else 1e-3

        for dim in valid_dims:
            tester = (
                Tester(self.Softmax(dim), inputs, dynamic_shapes=dynamic_shape)
                .export()
                .check_count({"torch.ops.aten.softmax": 1})
            )
            if enable_bf16:
                tester = tester.to_edge_transform_and_lower(
                    ToEdgeTransformAndLower(
                        partitioners=[XnnpackPartitioner(enable_bf16=True)]
                    )
                )
            else:
                tester = tester.to_edge_transform_and_lower()
            (
                tester.check_count(
                    {"torch.ops.higher_order.executorch_call_delegate": 1}
                )
                .check_not(["executorch_exir_dialects_edge__ops_aten__softmax_default"])
                .to_executorch()
                .serialize()
                .run_method_and_compare_outputs(num_runs=5, atol=atol, rtol=rtol)
            )

    def test_bf16_softmax(self):
        inputs = (torch.rand((3, 5, 7)).to(torch.bfloat16),)
        self._test_softmax(inputs, enable_bf16=True)

    def test_bf16_softmax_not_partitioned_without_enable_bf16(self):
        inputs = (torch.rand((3, 5, 7)).to(torch.bfloat16),)
        (
            Tester(self.Softmax(-1), inputs)
            .export()
            .check_count({"torch.ops.aten.softmax": 1})
            .to_edge_transform_and_lower()
            .check(["executorch_exir_dialects_edge__ops_aten__softmax_default"])
        )

    def test_fp16_softmax(self):
        inputs = (torch.rand((3, 5, 7)).to(torch.float16),)
        self._test_softmax(inputs)

    def test_fp32_softmax(self):
        inputs = (torch.rand((3, 5, 7)),)
        self._test_softmax(inputs)

    def test_fp32_softmax_unsupported(self):
        inputs = (torch.rand((3, 5, 7)),)

        # Dim can be either the last dimension index or -1 (last dimension),
        # as xnnpack only supports softmax on the last dimension.
        # This test validates the delegate does not attempt to delegate softmax
        # on any other dimension.
        invalid_dims = range(len(inputs) - 1)

        for dim in invalid_dims:
            (
                Tester(self.Softmax(dim), inputs)
                .export()
                .check_count({"torch.ops.aten.softmax": 1})
                .to_edge_transform_and_lower()
                # Should not be delegated
                .check(["executorch_exir_dialects_edge__ops_aten__softmax_default"])
            )
