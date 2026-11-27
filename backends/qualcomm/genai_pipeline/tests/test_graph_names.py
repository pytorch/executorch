# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.backends.qualcomm.genai_pipeline.graph_names import (
    DECODER_GRAPH_NAMES,
    GRAPH_FORWARD,
    GRAPH_KV_FORWARD,
    TOK_EMBEDDING_GRAPH_NAMES,
)


class TestGraphNames(unittest.TestCase):

    def test_names_match_legacy_decoder_constants(self):
        """Both graph-name lists equal the legacy flow's, order included."""
        from executorch.examples.qualcomm.oss_scripts.llama import decoder_constants

        self.assertEqual(DECODER_GRAPH_NAMES, decoder_constants.DECODER_GRAPH_NAMES)
        self.assertEqual(
            TOK_EMBEDDING_GRAPH_NAMES, decoder_constants.TOK_EMBEDDING_GRAPH_NAMES
        )

    def test_decode_graph_is_first(self):
        """Decode leads both lists: it is sliced first in kv mode and owns meta."""
        self.assertEqual(DECODER_GRAPH_NAMES[0], GRAPH_KV_FORWARD)
        self.assertIn(GRAPH_KV_FORWARD, TOK_EMBEDDING_GRAPH_NAMES[0])

    def test_single_graph_name_is_not_a_deployed_decoder_graph(self):
        """``GRAPH_FORWARD`` names the calibration / encoder graph, never a method."""
        self.assertNotIn(GRAPH_FORWARD, DECODER_GRAPH_NAMES)
        self.assertNotIn(GRAPH_FORWARD, TOK_EMBEDDING_GRAPH_NAMES)


if __name__ == "__main__":
    unittest.main()
