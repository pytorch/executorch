# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

from backends.qualcomm.genai_pipeline.configs.inference_input_config import (
    InferenceInputConfig,
)


class TestInferenceInputConfig(unittest.TestCase):

    def test_inference_options_default_factory(self):
        c1 = InferenceInputConfig(soc_model=MagicMock())
        c2 = InferenceInputConfig(soc_model=MagicMock())
        c1.inference_options["key"] = "val"
        self.assertEqual(c2.inference_options, {})

    def test_required_fields(self):
        with self.assertRaises(TypeError):
            InferenceInputConfig()


if __name__ == "__main__":
    unittest.main()
