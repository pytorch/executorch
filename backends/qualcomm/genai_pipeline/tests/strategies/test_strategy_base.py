# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.compilation_strategy import (
    CompilationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.inference.inference_strategy import (
    InferenceStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.quantization_strategy import (
    QuantizationStrategy,
)


class TestStrategyABCs(unittest.TestCase):

    def test_cannot_instantiate_quantization_strategy(self):
        with self.assertRaises(TypeError):
            QuantizationStrategy()

    def test_cannot_instantiate_compilation_strategy(self):
        with self.assertRaises(TypeError):
            CompilationStrategy()

    def test_cannot_instantiate_inference_strategy(self):
        with self.assertRaises(TypeError):
            InferenceStrategy()

    def test_subclass_must_implement_invoke(self):
        class BadQuantStrategy(QuantizationStrategy):
            pass

        with self.assertRaises(TypeError):
            BadQuantStrategy()


if __name__ == "__main__":
    unittest.main()
