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
from executorch.backends.qualcomm.genai_pipeline.graph_names import (
    GRAPH_FORWARD,
    GRAPH_KV_FORWARD,
)
from executorch.backends.qualcomm.genai_pipeline.graph_roles import (
    get_graph_role,
    GraphRole,
    is_deployed,
)


class TestGraphRoles(unittest.TestCase):

    def test_only_quantize_graphs_are_not_deployed(self):
        """The decoder calibration graph is dropped; deploy and shared graphs lower."""
        self.assertFalse(is_deployed(ARTIFACT_TEXT_DECODER, GRAPH_FORWARD))
        self.assertTrue(is_deployed(ARTIFACT_TEXT_DECODER, GRAPH_KV_FORWARD))
        self.assertIs(
            get_graph_role(ARTIFACT_VISION_ENCODER, GRAPH_FORWARD), GraphRole.SHARED
        )
        self.assertTrue(is_deployed(ARTIFACT_VISION_ENCODER, GRAPH_FORWARD))

    def test_rejects_an_unknown_component(self):
        """A component without roles cannot be routed by either stage."""
        with self.assertRaises(ValueError) as ctx:
            get_graph_role("not_a_component", GRAPH_FORWARD)

        self.assertIn("not_a_component", str(ctx.exception))

    def test_rejects_an_unknown_graph(self):
        """A graph name outside the table is a producer and table mismatch."""
        with self.assertRaises(ValueError) as ctx:
            get_graph_role(ARTIFACT_TEXT_DECODER, "not_a_graph")

        self.assertIn("not_a_graph", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
