# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester
from executorch.examples.models.mobilebert.modeling_mobilebert import MobileBertModel
from transformers import MobileBertConfig  # @manual


class TestMobilebert(unittest.TestCase):
    mobilebert = MobileBertModel(MobileBertConfig()).eval()
    example_inputs = (torch.tensor([[101, 7592, 1010, 2026, 3899, 2003, 10140, 102]]),)
    supported_ops = {
        "executorch_exir_dialects_edge__ops_aten_addmm_default",
        "executorch_exir_dialects_edge__ops_aten_add_Tensor",
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
        "executorch_exir_dialects_edge__ops_aten_sub_Tensor",
        "executorch_exir_dialects_edge__ops_aten_div_Tensor",
        "executorch_exir_dialects_edge__ops_aten_cat_default",
        "executorch_exir_dialects_edge__ops_aten_relu_default",
        "executorch_exir_dialects_edge__ops_aten_permute_copy_default",
        "executorch_exir_dialects_edge__ops_aten__softmax_default",
        "executorch_exir_dialects_edge__ops_aten_constant_pad_nd_default",
    }

    def test_fp32_mobilebert(self):
        (
            Tester(self.mobilebert, self.example_inputs)
            .export()
            .to_edge()
            .check(list(self.supported_ops))
            .partition()
            .check_not(list(self.supported_ops))
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(inputs=self.example_inputs)
        )
