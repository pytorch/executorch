# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the checkpoint key-renaming state-dict transforms."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from executorch.backends.qualcomm.genai_pipeline.source_transform.checkpoint_key_remap import (
    remap_gemma4_keys,
    strip_orig_mod_prefix,
)


class TestStripOrigModPrefix(unittest.TestCase):
    """``_orig_mod.`` is left on keys by torch.compile checkpoints."""

    def test_removes_prefix(self):
        result = strip_orig_mod_prefix(
            {"_orig_mod.layers.0.attention.wq.weight": torch.zeros(1)}
        )
        self.assertEqual(list(result), ["layers.0.attention.wq.weight"])

    def test_leaves_unprefixed_keys_alone(self):
        result = strip_orig_mod_prefix({"tok_embeddings.weight": torch.zeros(1)})
        self.assertEqual(list(result), ["tok_embeddings.weight"])


class TestRemapGemma4Keys(unittest.TestCase):
    """Gemma4 arrives pre-converted and only needs its keys renamed."""

    def test_delegates_to_gemma4_remap(self):
        state_dict = {"before": torch.zeros(1)}
        remapped = {"after": torch.zeros(1)}
        module = MagicMock()
        module.remap_keys.return_value = remapped

        with patch.dict(
            "sys.modules",
            {
                "executorch.examples.qualcomm.oss_scripts.gemma4.text_decoder.convert_weights": module
            },
        ):
            result = remap_gemma4_keys(state_dict)

        module.remap_keys.assert_called_once_with(state_dict)
        self.assertIs(result, remapped)


if __name__ == "__main__":
    unittest.main()
