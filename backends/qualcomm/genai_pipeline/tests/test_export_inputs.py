# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_TEXT_DECODER,
    ARTIFACT_VISION_ENCODER,
)
from executorch.backends.qualcomm.genai_pipeline.export_inputs import to_export_inputs
from executorch.backends.qualcomm.genai_pipeline.graph_names import (
    GRAPH_FORWARD,
    GRAPH_KV_FORWARD,
)

_DECODER_INPUTS = ("tok", ("mask_0", "mask_1"), "pos", ("k_0",), ("v_0",))


def _meta(use_kv_cache):
    """Create decoder metadata for the decode graph."""
    return {
        ARTIFACT_TEXT_DECODER: {GRAPH_KV_FORWARD: {"get_use_kv_cache": use_kv_cache}}
    }


class TestToExportInputs(unittest.TestCase):

    def test_flattens_decoder_inputs_with_kv_cache(self):
        """Masks, position IDs and caches are expanded into positional inputs."""
        inputs = to_export_inputs(
            ARTIFACT_TEXT_DECODER, GRAPH_KV_FORWARD, _DECODER_INPUTS, _meta(True)
        )

        self.assertEqual(inputs, ("tok", "mask_0", "mask_1", "pos", "k_0", "v_0"))

    def test_drops_position_ids_and_caches_without_kv_cache(self):
        """A graph without a KV cache exports tokens and masks only."""
        inputs = to_export_inputs(
            ARTIFACT_TEXT_DECODER, GRAPH_KV_FORWARD, _DECODER_INPUTS, _meta(False)
        )

        self.assertEqual(inputs, ("tok", "mask_0", "mask_1"))

    def test_passes_non_decoder_inputs_through(self):
        """Only the decoder has a nested input structure."""
        inputs = ("pixels",)

        self.assertIs(
            to_export_inputs(ARTIFACT_VISION_ENCODER, GRAPH_FORWARD, inputs, {}), inputs
        )

    def test_rejects_a_decoder_graph_without_the_kv_cache_flag(self):
        """Guessing the signature would export the wrong graph."""
        with self.assertRaises(ValueError) as ctx:
            to_export_inputs(
                ARTIFACT_TEXT_DECODER, GRAPH_KV_FORWARD, _DECODER_INPUTS, {}
            )

        self.assertIn("get_use_kv_cache", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
