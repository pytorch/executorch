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
from executorch.backends.qualcomm.genai_pipeline.configs.inference_output_config import (
    InferenceOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.pipeline_types import STAGE_INFERENCE
from executorch.backends.qualcomm.genai_pipeline.stages.inference_stage import (
    InferenceStage,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.inference.inference_strategy import (
    InferenceStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import (
    make_test_context,
)


class TestInferenceStage(unittest.TestCase):

    def test_name(self):
        mock_strategy = MagicMock(spec=InferenceStrategy)
        stage = InferenceStage(mock_strategy)
        self.assertEqual(stage.name, STAGE_INFERENCE)

    def test_invoke_delegates_to_strategy(self):
        mock_strategy = MagicMock(spec=InferenceStrategy)
        mock_strategy.invoke.return_value = InferenceOutputConfig(
            inference_results=["output"]
        )
        stage = InferenceStage(mock_strategy)
        context = make_test_context()
        input_config = InferenceInputConfig(soc_model=MagicMock())

        stage.invoke(context, input_config)

        mock_strategy.invoke.assert_called_once_with(context, input_config)


if __name__ == "__main__":
    unittest.main()
