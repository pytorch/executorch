# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest
from pathlib import Path

import torch
from executorch.backends.test.harness.stages import StageType
from executorch.backends.xnnpack.test.tester import Tester
from executorch.runtime import Runtime, Verification


class TestPrelu(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    class PReLU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.prelu = torch.nn.PReLU(num_parameters=5, init=0.2)

        def forward(self, x):
            a = self.prelu(x)
            return a

    def _test_prelu(self, module, inputs):
        (
            Tester(module, inputs)
            .export()
            .check_count({"torch.ops.aten.prelu.default": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                ["executorch_exir_dialects_edge__ops_aten__prelu_kernel_default"]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    @unittest.skip("XNNPACK Expects FP16 inputs but FP32 weights")
    def _test_fp16_prelu(self):
        module = self.PReLU().to(torch.float16)
        inputs = (torch.randn(1, 5, 3, 2).to(torch.float16),)
        self._test_prelu(module, inputs)

    def test_fp32_prelu(self):
        module = self.PReLU()
        inputs = (torch.randn(1, 5, 3, 2),)
        self._test_prelu(module, inputs)

    def test_fp32_prelu_file_load(self):
        """
        Make sure that PreLU doesn't free its weight buffer after load. It's a weird
        op that doesn't copy or pack its data, so we need to hold onto the buffer.
        Run specifically from a file to exercise the path.
        """
        module = self.PReLU()
        module.eval()
        x = torch.randn(1, 5, 3, 2)
        expected = module(x)

        tester = Tester(module, (x,))
        tester.export()
        tester.to_edge_transform_and_lower()
        tester.check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
        tester.to_executorch()
        tester.serialize()

        buf = tester.stages[StageType.SERIALIZE].artifact
        fd, pte_path = tempfile.mkstemp(suffix=".pte")
        try:
            os.write(fd, buf)
            os.close(fd)
            rt = Runtime.get()
            program = rt.load_program(Path(pte_path), verification=Verification.Minimal)
            method = program.load_method("forward")
            actual = method.execute((x,))[0]
            self.assertTrue(torch.allclose(expected, actual, atol=1e-5))
        finally:
            os.unlink(pte_path)
