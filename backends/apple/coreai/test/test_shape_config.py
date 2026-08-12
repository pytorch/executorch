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


class _TwoInputs(nn.Module):
    """Both inputs tied to one batch Dim, as a model with paired tensors is."""

    def forward(self, x, y):
        return x + y


def _shared_dim_ep():
    batch = torch.export.Dim("batch", min=2, max=64)
    return torch.export.export(
        _TwoInputs().eval(),
        (torch.randn(4, 8), torch.randn(4, 8)),
        dynamic_shapes={"x": {0: batch}, "y": {0: batch}},
    )


class _Elementwise(nn.Module):
    """No shape constraints, so both dims can be dynamic."""

    def forward(self, x):
        return x + x


def _two_dynamic_dims_ep():
    """Dynamic batch and sequence, as a transformer input has."""
    batch = torch.export.Dim("batch", min=2, max=64)
    seq = torch.export.Dim("seq", min=4, max=32)
    return torch.export.export(
        _Elementwise().eval(),
        (torch.randn(4, 8),),
        dynamic_shapes={"x": {0: batch, 1: seq}},
    )


def _capture_shape_configs(exported_program, input_enumerations):
    """Every shapes_config handed to the SDK during a full lowering."""
    from coreai.authoring import AIProgram

    captured = []
    orig = AIProgram.set_static_shape_config

    def _record(self, entrypoint, shapes_config):
        captured.append(shapes_config)
        return orig(self, entrypoint, shapes_config)

    AIProgram.set_static_shape_config = _record
    try:
        to_edge_transform_and_lower(
            exported_program,
            partitioner=[CoreAIPartitioner(input_enumerations=input_enumerations)],
        ).to_executorch()
    finally:
        AIProgram.set_static_shape_config = orig
    return captured


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


class InputEnumerationsValidationTest(unittest.TestCase):
    """User-supplied enumerations that cannot be honoured must say so."""

    def test_conflicting_values_for_one_symbol_are_rejected(self):
        """Two inputs tied to the same Dim cannot enumerate differently.

        They resolve to one export symbol, so the second list silently wins.
        """
        with self.assertRaises(ValueError):
            resolve_input_enumerations(_shared_dim_ep(), [{0: [2, 4]}, {0: [8, 16]}])

    def test_dim_index_out_of_range_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_input_enumerations(_dynamic_ep(), [{9: [2, 4]}])

    def test_empty_values_are_rejected(self):
        """An empty list makes the enumeration vanish rather than apply."""
        with self.assertRaises(ValueError):
            resolve_input_enumerations(_dynamic_ep(), [{0: []}])

    def test_values_outside_the_declared_range_are_rejected(self):
        """The exported Dim bounds the values the model can actually accept."""
        with self.assertRaises(ValueError):
            resolve_input_enumerations(_dynamic_ep(), [{0: [4, 10**6]}])


class MultiInputEnumerationsTest(unittest.TestCase):
    def test_inputs_sharing_a_symbol_are_enumerated_together(self):
        """One key per combination, naming every input it constrains.

        ``set_static_shape_config`` treats a key as a whole-graph
        specialization, so enumerating each input separately leaves the other
        unconstrained in every entry.
        """
        captured = _capture_shape_configs(_shared_dim_ep(), [{0: [2, 4, 8]}, None])
        self.assertTrue(captured, "set_static_shape_config was not called")
        entries = {k: v for cfg in captured for k, v in cfg.items()}
        self.assertEqual(
            len(entries), 3, f"expected one entry per value, got {sorted(entries)}"
        )
        for key, shapes in entries.items():
            with self.subTest(key):
                self.assertEqual(
                    set(shapes), {"x", "y"}, f"{key} leaves an input unconstrained"
                )
                self.assertEqual(len({tuple(s) for s in shapes.values()}), 1)

    def test_unenumerated_dynamic_dim_is_reported(self):
        """A second dynamic dim left un-enumerated cannot be made concrete.

        Without a check this reaches sympy as ``Cannot convert symbols to int``,
        naming neither the input nor the dim.
        """
        with self.assertRaisesRegex(Exception, "s1|dim|enumerat"):
            to_edge_transform_and_lower(
                _two_dynamic_dims_ep(),
                partitioner=[CoreAIPartitioner(input_enumerations=[{0: [2, 4]}])],
            ).to_executorch()


if __name__ == "__main__":
    unittest.main()
