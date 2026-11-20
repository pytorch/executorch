# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from executorch.backends.qualcomm.genai_pipeline.compilation.model_layout import (
    ModelLayout,
)
from executorch.backends.qualcomm.genai_pipeline.model_lookup import (
    get_model_arch,
    get_model_config,
    get_model_layout,
)


class TestGetModelLayout(unittest.TestCase):

    def test_most_models_use_the_default_layout(self):
        """A model registers nothing unless it differs from the default."""
        self.assertEqual(get_model_layout("llama3_2-1b_instruct"), ModelLayout())

    def test_gemma4_offsets_its_cross_decoder_layers(self):
        """Gemma4 splits its layers over two modules, so one is offset."""
        layout = get_model_layout("gemma4-e2b")

        self.assertEqual(layout.offset_layer_prefix, "model.cross_decoder.layers")
        self.assertIn(layout.offset_layer_prefix, layout.decoder_layer_prefixes)


class TestModelLookup(unittest.TestCase):

    def test_mixed_case_model_name_resolves_to_canonical_config(self):
        self.assertIs(
            get_model_config("StOrIeS260K"),
            get_model_config("stories260k"),
        )

    def test_get_model_arch_rejects_decoder_without_recipe(self):
        config = SimpleNamespace(quant_recipe=None)
        control_args = argparse.Namespace(
            embedding_quantize=None,
            model_mode="kv",
        )

        with patch(
            "executorch.backends.qualcomm.genai_pipeline.model_lookup.get_model_config",
            return_value=config,
        ):
            with self.assertRaisesRegex(ValueError, "KV-cache IO bit width"):
                get_model_arch("custom", control_args)


if __name__ == "__main__":
    unittest.main()
