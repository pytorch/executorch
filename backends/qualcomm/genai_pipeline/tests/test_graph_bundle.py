# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import unittest
from unittest.mock import MagicMock

import torch
from executorch.backends.qualcomm.genai_pipeline.graph_bundle import GraphBundle


def _make_bundle(**overrides) -> GraphBundle:
    """Create a GraphBundle with valid defaults."""
    kwargs = {
        "module": MagicMock(name="graph_module"),
        "inputs": (MagicMock(name="tokens"), MagicMock(name="attn_mask")),
    }
    kwargs.update(overrides)
    return GraphBundle(**kwargs)


class TestGraphBundleFields(unittest.TestCase):

    def test_module_and_inputs_are_required(self):
        """A bundle without a graph or its export signature is not constructible."""
        with self.assertRaises(TypeError):
            GraphBundle()

    def test_lowering_options_default_to_absent(self):
        """The optional lowering inputs default to None, and meta to empty."""
        bundle = _make_bundle()

        self.assertEqual(bundle.meta, {})
        self.assertIsNone(bundle.quant_io_dtypes)
        self.assertIsNone(bundle.modality_inputs)
        self.assertIsNone(bundle.executorch_config)

    def test_meta_is_not_shared_between_bundles(self):
        """Each bundle gets its own meta, since quantization mutates it in place."""
        first, second = _make_bundle(), _make_bundle()

        first.meta["get_logits_scale"] = 0.5

        self.assertEqual(second.meta, {})


class TestQuantIoDtypes(unittest.TestCase):
    """The tagger indexes both keys per node, so a partial mapping is invalid."""

    def test_both_dtypes_are_accepted(self):
        """The mapping quantization publishes carries both boundary dtypes."""
        quant_io_dtypes = {"kv_type": torch.uint8, "io_type": torch.uint16}

        bundle = _make_bundle(quant_io_dtypes=quant_io_dtypes)

        self.assertIs(bundle.quant_io_dtypes, quant_io_dtypes)

    def test_skipped_quantization_is_none_not_a_partial_mapping(self):
        """None is how a graph says its IO stays float32."""
        bundle = _make_bundle(quant_io_dtypes=None)

        self.assertIsNone(bundle.quant_io_dtypes)

    def test_partial_mapping_is_rejected(self):
        """One key alone would be a KeyError at lowering, so reject it here."""
        for partial in ({"kv_type": torch.uint8}, {"io_type": torch.uint16}, {}):
            with self.subTest(quant_io_dtypes=partial):
                with self.assertRaises(ValueError):
                    _make_bundle(quant_io_dtypes=partial)

    def test_unknown_key_is_rejected(self):
        """An unrecognised key means the producer and the tagger disagree."""
        with self.assertRaises(ValueError):
            _make_bundle(
                quant_io_dtypes={
                    "kv_type": torch.uint8,
                    "io_type": torch.uint16,
                    "logits_type": torch.uint16,
                }
            )

    def test_replace_is_validated_too(self):
        """``dataclasses.replace`` re-runs __post_init__, so it cannot bypass this."""
        bundle = _make_bundle(
            quant_io_dtypes={"kv_type": torch.uint8, "io_type": torch.uint16}
        )

        with self.assertRaises(ValueError):
            dataclasses.replace(bundle, quant_io_dtypes={"kv_type": torch.uint8})


class TestGraphBundleImmutability(unittest.TestCase):

    def test_fields_cannot_be_reassigned(self):
        """Frozen, so a consumer cannot repoint a bundle's fields in place."""
        bundle = _make_bundle()

        with self.assertRaises(dataclasses.FrozenInstanceError):
            bundle.module = MagicMock(name="other_module")

    def test_replace_returns_a_new_bundle_carrying_the_rest(self):
        """``dataclasses.replace`` is how a stage adds to a bundle."""
        bundle = _make_bundle()
        quant_io_dtypes = {"kv_type": torch.uint8, "io_type": torch.uint16}

        updated = dataclasses.replace(bundle, quant_io_dtypes=quant_io_dtypes)

        self.assertIs(updated.quant_io_dtypes, quant_io_dtypes)
        self.assertIs(updated.module, bundle.module)
        self.assertIs(updated.inputs, bundle.inputs)
        self.assertIsNone(bundle.quant_io_dtypes)


if __name__ == "__main__":
    unittest.main()
