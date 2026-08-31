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


class TestQuantizationInputConfig(unittest.TestCase):

    def test_extra_options_default_factory(self):
        c1 = QuantizationInputConfig(soc_model=MagicMock(), backend_type=MagicMock())
        c2 = QuantizationInputConfig(soc_model=MagicMock(), backend_type=MagicMock())
        c1.extra_options["key"] = "val"
        self.assertEqual(c2.extra_options, {})

    def test_required_fields(self):
        with self.assertRaises(TypeError):
            QuantizationInputConfig()

    def test_optional_fields_default_to_none(self):
        config = QuantizationInputConfig(
            soc_model=MagicMock(), backend_type=MagicMock()
        )
        self.assertIsNone(config.model_module)
        self.assertIsNone(config.example_inputs)
        self.assertIsNone(config.calibration_data)
        self.assertIsNone(config.training_data)
        self.assertIsNone(config.quant_recipe)

    def test_training_data_carries_qat_dataset(self):
        training_data = [("features", "labels")]
        config = QuantizationInputConfig(
            soc_model=MagicMock(),
            backend_type=MagicMock(),
            training_data=training_data,
        )
        self.assertIs(config.training_data, training_data)


if __name__ == "__main__":
    unittest.main()
