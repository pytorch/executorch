# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.backends.qualcomm.genai_pipeline.configs.model_preparation_input_config import (
    ModelPreparationInputConfig,
)


class TestModelPreparationInputConfig(unittest.TestCase):

    def test_extra_options_default_factory(self):
        c1 = ModelPreparationInputConfig(model_name="model_a", soc_model="SM8750")
        c2 = ModelPreparationInputConfig(model_name="model_b", soc_model="SM8650")
        c1.extra_options["key"] = "val"
        self.assertEqual(c2.extra_options, {})

    def test_required_fields(self):
        with self.assertRaises(TypeError):
            ModelPreparationInputConfig()


if __name__ == "__main__":
    unittest.main()
