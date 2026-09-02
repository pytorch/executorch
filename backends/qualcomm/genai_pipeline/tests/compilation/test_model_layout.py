# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.backends.qualcomm.genai_pipeline.compilation import model_layout
from executorch.backends.qualcomm.genai_pipeline.compilation.model_layout import (
    get_model_layout,
    ModelLayout,
    register_model_layout,
)

_HF_MODEL = "test-hf-model"


class TestModelLayout(unittest.TestCase):

    def test_an_unregistered_model_gets_the_default_layout(self):
        """Only a model that differs from the default registers a layout."""
        self.assertEqual(get_model_layout("llama3_2-1b_instruct"), ModelLayout())

    def test_gemma4_registers_its_two_decoder_module_paths(self):
        """Gemma4's layers span a self-decoder and a cross-decoder."""
        layout = get_model_layout("gemma4-e2b")

        self.assertIn("model.self_decoder.layers", layout.decoder_layer_prefixes)
        self.assertIn("model.cross_decoder.layers", layout.decoder_layer_prefixes)

    def test_a_registered_layout_is_returned(self):
        """A new model adds an entry instead of a model-name conditional."""
        self.addCleanup(model_layout._LAYOUTS.pop, _HF_MODEL, None)
        layout = ModelLayout(decoder_layer_prefixes=("model.layers",))

        register_model_layout(_HF_MODEL, layout)

        self.assertIs(get_model_layout(_HF_MODEL), layout)

    def test_rejects_a_conflicting_registration(self):
        """Two layouts for one model would make lowering order-dependent."""
        self.addCleanup(model_layout._LAYOUTS.pop, _HF_MODEL, None)
        register_model_layout(_HF_MODEL, ModelLayout(decoder_layer_prefixes=("a",)))

        with self.assertRaises(ValueError):
            register_model_layout(_HF_MODEL, ModelLayout(decoder_layer_prefixes=("b",)))


if __name__ == "__main__":
    unittest.main()
