# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from executorch.backends.qualcomm.genai_pipeline.model_lookup import (
    get_model_arch,
    get_model_config,
)


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
