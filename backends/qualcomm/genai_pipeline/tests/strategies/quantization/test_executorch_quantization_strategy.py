# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

from executorch.backends.qualcomm.genai_pipeline.configs.quantization_input_config import (
    QuantizationInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.executorch_quantization_strategy import (
    ExecuTorchQuantizationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.quantization_strategy import (
    QuantizationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import (
    make_test_context,
)


class TestExecuTorchQuantizationStrategy(unittest.TestCase):

    def test_is_quantization_strategy(self):
        strategy = ExecuTorchQuantizationStrategy()
        self.assertIsInstance(strategy, QuantizationStrategy)

    def test_invoke_raises_not_implemented(self):
        strategy = ExecuTorchQuantizationStrategy()
        with self.assertRaises(NotImplementedError):
            strategy.invoke(
                make_test_context(),
                QuantizationInputConfig(
                    soc_model=MagicMock(), backend_type=MagicMock()
                ),
            )


if __name__ == "__main__":
    unittest.main()
