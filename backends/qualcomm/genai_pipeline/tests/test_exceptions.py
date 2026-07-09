# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.backends.qualcomm.genai_pipeline.exceptions import (
    ConfigValidationError,
    EngineNotAvailableError,
    PipelineError,
    StageError,
)


class TestExceptionHierarchy(unittest.TestCase):

    def test_all_exceptions_inherit_from_pipeline_error(self):
        self.assertIsInstance(StageError("s", "m"), PipelineError)
        self.assertIsInstance(ConfigValidationError("m"), PipelineError)
        self.assertIsInstance(EngineNotAvailableError("e"), PipelineError)


class TestStageError(unittest.TestCase):

    def test_message_includes_stage_name(self):
        error = StageError("quantization", "model failed")
        self.assertIn("[quantization]", str(error))
        self.assertIn("model failed", str(error))

    def test_original_exception_chained(self):
        cause = RuntimeError("out of memory")
        error = StageError("quantization", "failed", original_exception=cause)
        self.assertIs(error.original_exception, cause)
        self.assertIn("RuntimeError", str(error))
        self.assertIn("out of memory", str(error))


class TestEngineNotAvailableError(unittest.TestCase):

    def test_custom_message(self):
        error = EngineNotAvailableError("test_engine", "SDK not installed")
        self.assertIn("test_engine", str(error))
        self.assertIn("SDK not installed", str(error))


if __name__ == "__main__":
    unittest.main()
