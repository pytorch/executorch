# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

from backends.qualcomm.genai_pipeline.configs.quantization_input_config import (
    QuantizationInputConfig,
)


class TestQuantizationInputConfig(unittest.TestCase):

    def test_extra_options_default_factory(self):
        c1 = QuantizationInputConfig(soc_model=MagicMock(), backend_type=MagicMock())
        c2 = QuantizationInputConfig(soc_model=MagicMock(), backend_type=MagicMock())
        c1.extra_options["key"] = "val"
        self.assertEqual(c2.extra_options, {})

    def test_required_fields(self):
        with self.assertRaises(TypeError):
            QuantizationInputConfig()


if __name__ == "__main__":
    unittest.main()
