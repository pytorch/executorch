# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

from executorch.backends.qualcomm.genai_pipeline.configs.inference_input_config import (
    InferenceInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.inference.executorch_inference_strategy import (
    ExecuTorchInferenceStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.inference.inference_strategy import (
    InferenceStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import (
    make_test_context,
)


class TestExecuTorchInferenceStrategy(unittest.TestCase):

    def test_is_inference_strategy(self):
        strategy = ExecuTorchInferenceStrategy()
        self.assertIsInstance(strategy, InferenceStrategy)

    def test_invoke_raises_not_implemented(self):
        strategy = ExecuTorchInferenceStrategy()
        with self.assertRaises(NotImplementedError):
            strategy.invoke(
                make_test_context(),
                InferenceInputConfig(soc_model=MagicMock()),
            )


if __name__ == "__main__":
    unittest.main()
