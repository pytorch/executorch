# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch


class TestCustomQC4WConvert(unittest.TestCase):
    def setUp(self):
        torch.ops.load_library("//executorch/extension/aot_util:aot_util")

    def test_convert(self):
        def _ref_output(inp):
            oc, ic = inp.shape
            if ic % 2 != 0:
                raise ValueError("Number of input channels not divisible by 2.")
            ric = (ic + 1) // 2
            result = torch.zeros([oc, ric], dtype=torch.uint8)
            for o in range(oc):
                for i in range(ric):
                    j = 2 * i
                    result[o][i] = inp[o][j]
                    result[o][i] += inp[o][j + 1] << 4
            return result

        inp = torch.randint(low=0, high=15, size=(20, 42), dtype=torch.uint8)
        result = torch.ops.xnnpack.convert_to_qc4w(inp)
        ref_result = _ref_output(inp)
        assert torch.equal(result, ref_result), "Outputs dont match"

    def test_convert_throws(self):
        inp = torch.randint(low=0, high=15, size=(20, 41), dtype=torch.uint8)
        exception_thrown = False
        # Because for some reason self.assertRaises does not work
        # and didnt try to debug
        try:
            torch.ops.xnnpack.convert_to_qc4w(inp)
        except:
            exception_thrown = True
        self.assertTrue(exception_thrown)

        inp = torch.rand((20, 41))
        exception_thrown = False
        # Because for some reason self.assertRaises does not work
        # and didnt try to debug
        try:
            torch.ops.xnnpack.convert_to_qc4w(inp)
        except:
            exception_thrown = True
        self.assertTrue(exception_thrown)
