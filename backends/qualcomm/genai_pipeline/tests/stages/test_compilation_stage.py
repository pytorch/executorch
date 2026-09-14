# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

from executorch.backends.qualcomm.genai_pipeline.configs.compilation_input_config import (
    CompilationInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.compilation_output_config import (
    CompilationOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.pipeline_types import STAGE_COMPILATION
from executorch.backends.qualcomm.genai_pipeline.stages.compilation_stage import (
    CompilationStage,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.compilation_strategy import (
    CompilationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import (
    make_test_context,
    TEST_PTE_PATH,
)


class TestCompilationStage(unittest.TestCase):

    def test_name(self):
        mock_strategy = MagicMock(spec=CompilationStrategy)
        stage = CompilationStage(mock_strategy)
        self.assertEqual(stage.name, STAGE_COMPILATION)

    def test_invoke_delegates_to_strategy(self):
        mock_strategy = MagicMock(spec=CompilationStrategy)
        mock_strategy.invoke.return_value = CompilationOutputConfig(
            artifact_paths=[TEST_PTE_PATH]
        )
        stage = CompilationStage(mock_strategy)
        context = make_test_context()
        input_config = CompilationInputConfig(
            soc_model=MagicMock(), backend_type=MagicMock()
        )

        stage.invoke(context, input_config)

        mock_strategy.invoke.assert_called_once_with(context, input_config)


if __name__ == "__main__":
    unittest.main()
