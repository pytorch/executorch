# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

from executorch.backends.qualcomm.genai_pipeline.engine_proxy import EngineProxy
from executorch.backends.qualcomm.genai_pipeline.pipeline_types import (
    EngineType,
    STAGE_COMPILATION,
    STAGE_INFERENCE,
    STAGE_MODEL_PREPARATION,
    STAGE_QUANTIZATION,
)


class TestEngineProxy(unittest.TestCase):

    def test_full_executorch_workflow(self):
        proxy = EngineProxy(
            {
                STAGE_MODEL_PREPARATION: EngineType.EXECUTORCH,
                STAGE_QUANTIZATION: EngineType.EXECUTORCH,
                STAGE_COMPILATION: EngineType.EXECUTORCH,
                STAGE_INFERENCE: EngineType.EXECUTORCH,
            },
            backend_type=MagicMock(name="kHtpBackend"),
        )
        self.assertEqual(
            proxy.get_engine(STAGE_MODEL_PREPARATION), EngineType.EXECUTORCH
        )
        self.assertEqual(proxy.get_engine(STAGE_QUANTIZATION), EngineType.EXECUTORCH)
        self.assertEqual(proxy.get_engine(STAGE_COMPILATION), EngineType.EXECUTORCH)
        self.assertEqual(proxy.get_engine(STAGE_INFERENCE), EngineType.EXECUTORCH)

    def test_default_engine_is_executorch(self):
        proxy = EngineProxy({}, backend_type=MagicMock(name="kHtpBackend"))
        self.assertEqual(
            proxy.get_engine(STAGE_MODEL_PREPARATION), EngineType.EXECUTORCH
        )
        self.assertEqual(proxy.get_engine(STAGE_QUANTIZATION), EngineType.EXECUTORCH)
        self.assertEqual(proxy.get_engine(STAGE_COMPILATION), EngineType.EXECUTORCH)
        self.assertEqual(proxy.get_engine(STAGE_INFERENCE), EngineType.EXECUTORCH)

    def test_backend_type_is_stored(self):
        backend = MagicMock(name="kHtpBackend")
        proxy = EngineProxy(
            {STAGE_INFERENCE: EngineType.EXECUTORCH}, backend_type=backend
        )
        self.assertIs(proxy.backend_type, backend)

    def test_stage_engines_returns_copy(self):
        proxy = EngineProxy(
            {STAGE_INFERENCE: EngineType.EXECUTORCH},
            backend_type=MagicMock(name="kHtpBackend"),
        )
        engines = proxy.stage_engines
        engines[STAGE_INFERENCE] = None
        self.assertEqual(proxy.get_engine(STAGE_INFERENCE), EngineType.EXECUTORCH)

    def test_invalid_stage_name_raises(self):
        with self.assertRaises(ValueError) as cm:
            EngineProxy(
                {"invalid_stage": EngineType.EXECUTORCH},
                backend_type=MagicMock(name="kHtpBackend"),
            )
        self.assertIn("Unknown stage", str(cm.exception))
        self.assertIn("invalid_stage", str(cm.exception))

    def test_empty_stage_engines(self):
        proxy = EngineProxy({}, backend_type=MagicMock(name="kHtpBackend"))
        self.assertEqual(proxy.get_engine(STAGE_QUANTIZATION), EngineType.EXECUTORCH)
        self.assertEqual(proxy.get_engine(STAGE_INFERENCE), EngineType.EXECUTORCH)


if __name__ == "__main__":
    unittest.main()
