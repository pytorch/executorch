# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestUnsqueeze(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    class Unsqueeze(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            y = x + x
            z = torch.unsqueeze(y, self.dim)
            return z

    def _test_unsqueeze(self, inputs, dim):
        (
            Tester(self.Unsqueeze(dim), inputs)
            .export()
            .check_count({"torch.ops.aten.unsqueeze.default": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_unsqueeze_copy_default"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_unsqueeze_dim_0(self):
        """Test unsqueeze at first dimension (dim=0)"""
        inputs = (torch.randn(4, 4),)
        self._test_unsqueeze(inputs, dim=0)

    def test_fp32_unsqueeze_dim_1(self):
        """Test unsqueeze at middle dimension (dim=1)"""
        inputs = (torch.randn(4, 4),)
        self._test_unsqueeze(inputs, dim=1)

    def test_fp32_unsqueeze_dim_last(self):
        """Test unsqueeze at last dimension (dim=-1)"""
        inputs = (torch.randn(4, 4),)
        self._test_unsqueeze(inputs, dim=-1)

    def test_fp32_unsqueeze_dim_negative(self):
        """Test unsqueeze with negative dimension index"""
        inputs = (torch.randn(4, 4),)
        self._test_unsqueeze(inputs, dim=-2)

    def test_fp32_unsqueeze_1d_tensor(self):
        """Test unsqueeze on a 1D tensor"""
        inputs = (torch.randn(8),)
        self._test_unsqueeze(inputs, dim=0)

    def test_fp32_unsqueeze_1d_tensor_last(self):
        """Test unsqueeze on a 1D tensor at last position"""
        inputs = (torch.randn(8),)
        self._test_unsqueeze(inputs, dim=1)

    def test_fp32_unsqueeze_3d_tensor(self):
        """Test unsqueeze on a 3D tensor"""
        inputs = (torch.randn(2, 3, 4),)
        self._test_unsqueeze(inputs, dim=2)

    def test_fp32_unsqueeze_4d_tensor(self):
        """Test unsqueeze on a 4D tensor"""
        inputs = (torch.randn(1, 2, 3, 4),)
        self._test_unsqueeze(inputs, dim=2)

    def test_fp16_unsqueeze(self):
        """Test FP16 unsqueeze"""
        inputs = (torch.randn(4, 4).to(torch.float16),)
        self._test_unsqueeze(inputs, dim=1)

    def test_fp16_unsqueeze_dim_0(self):
        """Test FP16 unsqueeze at first dimension"""
        inputs = (torch.randn(4, 4).to(torch.float16),)
        self._test_unsqueeze(inputs, dim=0)

    def test_fp16_unsqueeze_dim_negative(self):
        """Test FP16 unsqueeze with negative dimension"""
        inputs = (torch.randn(4, 4).to(torch.float16),)
        self._test_unsqueeze(inputs, dim=-1)
