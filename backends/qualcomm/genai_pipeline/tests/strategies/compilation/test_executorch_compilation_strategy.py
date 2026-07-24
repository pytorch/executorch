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
from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.compilation_strategy import (
    CompilationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.executorch_compilation_strategy import (
    ExecuTorchCompilationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import (
    make_test_context,
)


class TestExecuTorchCompilationStrategy(unittest.TestCase):

    def test_is_compilation_strategy(self):
        strategy = ExecuTorchCompilationStrategy()
        self.assertIsInstance(strategy, CompilationStrategy)

    def test_invoke_raises_not_implemented(self):
        strategy = ExecuTorchCompilationStrategy()
        with self.assertRaises(NotImplementedError):
            strategy.invoke(
                make_test_context(),
                CompilationInputConfig(soc_model=MagicMock(), backend_type=MagicMock()),
            )


if __name__ == "__main__":
    unittest.main()
