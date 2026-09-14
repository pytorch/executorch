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
from executorch.backends.qualcomm.genai_pipeline.configs.quantization_output_config import (
    QuantizationOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.pipeline_types import (
    STAGE_QUANTIZATION,
)
from executorch.backends.qualcomm.genai_pipeline.stages.quantization_stage import (
    QuantizationStage,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.quantization_strategy import (
    QuantizationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import (
    make_test_context,
)


class TestQuantizationStage(unittest.TestCase):

    def test_name(self):
        mock_strategy = MagicMock(spec=QuantizationStrategy)
        stage = QuantizationStage(mock_strategy)
        self.assertEqual(stage.name, STAGE_QUANTIZATION)

    def test_invoke_delegates_to_strategy(self):
        mock_strategy = MagicMock(spec=QuantizationStrategy)
        mock_strategy.invoke.return_value = QuantizationOutputConfig(
            quantized_model="quantized"
        )
        stage = QuantizationStage(mock_strategy)
        context = make_test_context()
        input_config = QuantizationInputConfig(
            soc_model=MagicMock(), backend_type=MagicMock()
        )

        stage.invoke(context, input_config)

        mock_strategy.invoke.assert_called_once_with(context, input_config)


if __name__ == "__main__":
    unittest.main()
