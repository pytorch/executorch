# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from executorch.backends.apple.coreai.compiler.enumerated_shapes import (
    resolve_input_enumerations,
)
from executorch.backends.apple.coreai.partition.partitioner import CoreAIPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.lowered_backend_module import executorch_call_delegate


def _model():
    return nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8)).eval()


def _dynamic_ep():
    batch = torch.export.Dim("batch", min=1, max=64)
    return torch.export.export(
        _model(), (torch.randn(2, 8),), dynamic_shapes={"input": {0: batch}}
    )


def _static_ep():
    return torch.export.export(_model(), (torch.randn(2, 8),))


class ResolveInputEnumerationsTest(unittest.TestCase):
    def test_maps_dynamic_dim_to_symbol(self):
        resolved = resolve_input_enumerations(_dynamic_ep(), [{0: [4, 16, 32]}])
        self.assertEqual(len(resolved), 1)
        self.assertEqual(list(resolved.values()), [[4, 16, 32]])
        # key is the export symbol name (e.g. "s31")
        (sym,) = resolved.keys()
        self.assertTrue(sym.startswith("s"))

    def test_static_dim_raises(self):
        with self.assertRaises(ValueError):
            resolve_input_enumerations(_static_ep(), [{0: [4, 16]}])

    def test_no_enumerations_is_none(self):
        self.assertIsNone(resolve_input_enumerations(_dynamic_ep(), None))


class InputEnumerationsLoweringTest(unittest.TestCase):
    def _lower(self, input_enumerations):
        return to_edge_transform_and_lower(
            _dynamic_ep(),
            partitioner=[CoreAIPartitioner(input_enumerations=input_enumerations)],
        )

    def test_lowers_to_executorch(self):
        lowered = self._lower([{0: [4, 16, 32]}])
        gm = lowered.exported_program().graph_module
        self.assertTrue(
            any(
                n.op == "call_function" and n.target is executorch_call_delegate
                for n in gm.graph.nodes
            )
        )
        self.assertGreater(len(bytes(lowered.to_executorch().buffer)), 0)

    def test_shapes_are_substituted_at_boundary(self):
        # Verify the ET-input enumeration propagates + substitutes into the
        # delegate boundary's symbolic shape (s31, 8) -> (4,8)/(16,8)/(32,8).
        from coreai.authoring import AIProgram

        captured = []
        orig = AIProgram.set_static_shape_config

        def _record(self, entrypoint, shapes_config):
            captured.append(shapes_config)
            return orig(self, entrypoint, shapes_config)

        AIProgram.set_static_shape_config = _record
        try:
            self._lower([{0: [4, 16, 32]}]).to_executorch()
        finally:
            AIProgram.set_static_shape_config = orig

        self.assertTrue(captured, "set_static_shape_config was not called")
        shapes = {
            tuple(next(iter(entry.values())))
            for cfg in captured
            for entry in cfg.values()
        }
        self.assertEqual(shapes, {(4, 8), (16, 8), (32, 8)})

    def test_static_dim_fails_lowering(self):
        with self.assertRaisesRegex(Exception, "static"):
            to_edge_transform_and_lower(
                _static_ep(),
                partitioner=[CoreAIPartitioner(input_enumerations=[{0: [4, 16]}])],
            )


if __name__ == "__main__":
    unittest.main()
