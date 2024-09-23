# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.backends.xnnpack.test.tester import Tester
from executorch.examples.models.llama2.model import Llama2Model


class TestLlama2ETExample(unittest.TestCase):
    def test_f32(self):
        self._test()

    def test_f16(self):
        self._test(torch.float16)

    # TODO - dynamic shape

    def _test(self, dtype: torch.dtype = torch.float):
        assert dtype in [
            torch.float,
            torch.float16,
        ], f"Only fp32 and fp16 are supported, but got dtype: {dtype}"

        llama2 = Llama2Model()
        model = llama2.get_eager_model().to(dtype)

        # Only convert fp32 inputs to dtype
        example_inputs = tuple(
            tensor.to(dtype) if tensor.dtype == torch.float32 else tensor
            for tensor in llama2.get_example_inputs()
        )

        (
            Tester(model, example_inputs)
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(atol=5e-2, inputs=example_inputs)
        )
