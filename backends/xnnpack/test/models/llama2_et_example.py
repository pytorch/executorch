# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.backends.xnnpack.test.tester import Tester
from executorch.examples.models.llama2.model import Llama2Model


class TestLlama2ETExample(unittest.TestCase):
    llama2 = Llama2Model()
    model = llama2.get_eager_model()
    example_inputs = llama2.get_example_inputs()

    # TODO - dynamic shape

    def test_fp32(self):
        (
            Tester(self.model, self.example_inputs)
            .export()
            .to_edge()
            .partition()
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )
