# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch
from executorch.backends.xnnpack._passes.convert_squeeze_to_view_pass import (
    ConvertSqueezeToViewPass,
)
from executorch.backends.xnnpack.test.tester import RunPasses, Tester
from executorch.exir.dialects._ops import ops as exir_ops


class TestConvertSqueezeToView(unittest.TestCase):
    PassStage = RunPasses([ConvertSqueezeToViewPass])

    class SqueezeModel(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            return torch.squeeze(x, self.dim)

    class UnsqueezeModel(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            return torch.unsqueeze(x, self.dim)

    def test_fp32_convert_squeeze_to_view(self):
        inputs = (torch.randn(1, 2, 1, 4, 1),)
        squeeze_dims = (0, 2, 4)

        for dims in squeeze_dims:
            (
                Tester(self.SqueezeModel(dims), inputs)
                .export()
                .to_edge()
                .check_node_count(
                    {
                        exir_ops.edge.aten.squeeze_copy.dims: 1,
                    }
                )
                .run_passes(self.PassStage)
                .check_node_count(
                    {
                        exir_ops.edge.aten.squeeze_copy.dims: 0,
                        exir_ops.edge.aten.view_copy.default: 1,
                    }
                )
                .run_method_and_compare_outputs()
            )

    def test_fp32_convert_unsqueeze_to_view(self):
        inputs = (torch.randn(1, 2, 4),)

        for dim in range(len(inputs[0].shape)):
            (
                Tester(self.UnsqueezeModel(dim), inputs)
                .export()
                .to_edge()
                .check_node_count(
                    {
                        exir_ops.edge.aten.unsqueeze_copy.default: 1,
                    }
                )
                .run_passes(self.PassStage)
                .check_node_count(
                    {
                        exir_ops.edge.aten.unsqueeze_copy.default: 0,
                        exir_ops.edge.aten.view_copy.default: 1,
                    }
                )
                .run_method_and_compare_outputs()
            )
