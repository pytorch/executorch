# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import unittest

import torch
from executorch.backends.xnnpack._passes.decompose_cat import DecomposeConcatenate
from executorch.backends.xnnpack.test.tester import RunPasses, Tester


class TestDecomposeCatPass(unittest.TestCase):
    PassStage = RunPasses([DecomposeConcatenate])
    cat_name = "executorch_exir_dialects_edge__ops_aten_cat_default"

    class Cat(torch.nn.Module):
        def forward(self, *args):
            xs = [*args]
            x = torch.cat(xs)
            return x + x  # Quantize by propagation.

    def test_cat_gt_5(self):
        inputs = [
            torch.randn(1, 2, 3),
        ]
        for num_inputs in range(6, 10):
            inputs = []
            for _ in range(num_inputs):
                inputs.append(torch.randn(1, 2, 3))

            num_cats = int(len(inputs) > 5)
            num_cats += math.ceil((len(inputs) - 5) / 4)
            (
                Tester(self.Cat(), tuple(inputs))
                .export()
                .to_edge()
                .check_count({self.cat_name: 1})
                .run_passes(self.PassStage)
                .check_count({self.cat_name: num_cats})
                .run_method_and_compare_outputs()
            )

    def test_cat_gt_10(self):
        inputs = [
            torch.randn(1, 2, 3),
        ]
        for num_inputs in [11, 16, 18]:
            inputs = []
            for _ in range(num_inputs):
                inputs.append(torch.randn(1, 2, 3))

            num_cats = int(len(inputs) > 5)
            num_cats += math.ceil((len(inputs) - 5) / 4)
            (
                Tester(self.Cat(), tuple(inputs))
                .export()
                .to_edge()
                .check_count({self.cat_name: 1})
                .run_passes(self.PassStage)
                .check_count({self.cat_name: num_cats})
                .run_method_and_compare_outputs()
            )

    def test_qs8_cat_gt_5(self):
        inputs = [
            torch.randn(1, 2, 3),
        ]
        for num_inputs in range(6, 10):
            inputs = []
            for _ in range(num_inputs):
                inputs.append(torch.randn(1, 2, 3))

            num_cats = int(len(inputs) > 5)
            num_cats += math.ceil((len(inputs) - 5) / 4)
            (
                Tester(self.Cat(), tuple(inputs))
                .quantize()
                .export()
                .to_edge()
                .check_count({self.cat_name: 1})
                .run_passes(self.PassStage)
                .check_count({self.cat_name: num_cats})
                .run_method_and_compare_outputs()
            )

    def test_qs8_cat_gt_10(self):
        inputs = [
            torch.randn(1, 2, 3),
        ]
        for num_inputs in [11, 16, 18]:
            inputs = []
            for _ in range(num_inputs):
                inputs.append(torch.randn(1, 2, 3))

            num_cats = int(len(inputs) > 5)
            num_cats += math.ceil((len(inputs) - 5) / 4)
            (
                Tester(self.Cat(), tuple(inputs))
                .quantize()
                .export()
                .to_edge()
                .check_count({self.cat_name: 1})
                .run_passes(self.PassStage)
                .check_count({self.cat_name: num_cats})
                .run_method_and_compare_outputs()
            )
