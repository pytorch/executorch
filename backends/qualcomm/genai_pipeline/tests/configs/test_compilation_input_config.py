# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from pathlib import Path
from unittest.mock import MagicMock

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_TEXT_DECODER,
)
from executorch.backends.qualcomm.genai_pipeline.configs.compilation_input_config import (
    CompilationInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.graph_bundle import GraphBundle
from executorch.backends.qualcomm.genai_pipeline.graph_names import (
    DECODER_GRAPH_NAMES,
    GRAPH_PREFILL_FORWARD,
)


class TestCompilationInputConfig(unittest.TestCase):

    def test_artifact_dir_default(self):
        config = CompilationInputConfig(soc_model=MagicMock(), backend_type=MagicMock())
        self.assertEqual(config.artifact_dir, Path("."))

    def test_required_fields(self):
        with self.assertRaises(TypeError):
            CompilationInputConfig()

    def test_optional_fields_default_to_none(self):
        config = CompilationInputConfig(soc_model=MagicMock(), backend_type=MagicMock())
        self.assertIsNone(config.model)
        self.assertIsNone(config.example_inputs)
        self.assertIsNone(config.compile_specs)
        self.assertIsNone(config.graphs)

    def test_example_inputs_carries_export_signature(self):
        example_inputs = (MagicMock(name="tokens"), MagicMock(name="attn_mask"))
        config = CompilationInputConfig(
            soc_model=MagicMock(),
            backend_type=MagicMock(),
            example_inputs=example_inputs,
        )
        self.assertIs(config.example_inputs, example_inputs)

    def test_graphs_are_addressable_by_component_then_graph_name(self):
        """One .pte per component, one method per graph within it."""
        decoder = {
            name: GraphBundle(module=MagicMock(name=name), inputs=(MagicMock(),))
            for name in DECODER_GRAPH_NAMES
        }
        config = CompilationInputConfig(
            soc_model=MagicMock(),
            backend_type=MagicMock(),
            graphs={ARTIFACT_TEXT_DECODER: decoder},
        )
        self.assertIs(
            config.graphs[ARTIFACT_TEXT_DECODER][GRAPH_PREFILL_FORWARD],
            decoder[GRAPH_PREFILL_FORWARD],
        )

    def test_num_sharding_defaults_to_one(self):
        """An unsharded graph must not request multiple contexts by default."""
        config = CompilationInputConfig(
            soc_model=MagicMock(),
            backend_type=MagicMock(),
        )
        self.assertEqual(config.num_sharding, 1)


if __name__ == "__main__":
    unittest.main()
