# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_TEXT_DECODER,
    ARTIFACT_VISION_ENCODER,
)
from executorch.backends.qualcomm.genai_pipeline.graph_names import (
    GRAPH_FORWARD,
    GRAPH_KV_FORWARD,
)
from executorch.backends.qualcomm.genai_pipeline.unquantized_bundles import (
    build_unquantized_bundles,
)


def _make_inputs():
    """Create model-preparation outputs for a decoder and a vision encoder."""
    model_module = {
        ARTIFACT_TEXT_DECODER: MagicMock(name="decoder"),
        ARTIFACT_VISION_ENCODER: MagicMock(name="encoder"),
    }
    example_inputs = {
        ARTIFACT_TEXT_DECODER: {
            GRAPH_FORWARD: ("calib", ("calib_mask",), "pos", ("k",), ("v",)),
            GRAPH_KV_FORWARD: ("tok", ("mask",), "pos", ("k",), ("v",)),
        },
        ARTIFACT_VISION_ENCODER: {GRAPH_FORWARD: ("pixels",)},
    }
    meta = {
        ARTIFACT_TEXT_DECODER: {
            GRAPH_FORWARD: {"get_use_kv_cache": False},
            GRAPH_KV_FORWARD: {"get_use_kv_cache": True},
        },
    }
    return model_module, example_inputs, meta


class TestBuildUnquantizedBundles(unittest.TestCase):

    def test_drops_the_decoder_calibration_graph(self):
        """A quantize-only graph has no use without quantization."""
        graphs = build_unquantized_bundles(*_make_inputs())

        self.assertEqual(set(graphs[ARTIFACT_TEXT_DECODER]), {GRAPH_KV_FORWARD})
        self.assertEqual(set(graphs[ARTIFACT_VISION_ENCODER]), {GRAPH_FORWARD})

    def test_bundles_the_flattened_export_signature(self):
        """Lowering re-exports with the inputs, so they must be flat."""
        graphs = build_unquantized_bundles(*_make_inputs())

        self.assertEqual(
            graphs[ARTIFACT_TEXT_DECODER][GRAPH_KV_FORWARD].inputs,
            ("tok", "mask", "pos", "k", "v"),
        )

    def test_marks_every_bundle_unquantized_without_dtypes(self):
        """Compilation reads the flag to request FP16."""
        graphs = build_unquantized_bundles(*_make_inputs())

        for component in graphs.values():
            for bundle in component.values():
                self.assertFalse(bundle.quantized)
                self.assertIsNone(bundle.quant_io_dtypes)

    def test_rejects_a_component_without_a_module(self):
        """Inputs without a module cannot be lowered."""
        model_module, example_inputs, meta = _make_inputs()
        del model_module[ARTIFACT_VISION_ENCODER]

        with self.assertRaises(ValueError):
            build_unquantized_bundles(model_module, example_inputs, meta)


if __name__ == "__main__":
    unittest.main()
