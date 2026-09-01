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
from executorch.backends.apple.coreai.partition.partitioner import (
    CoreAIPartitioner,
    do_not_delegate,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.lowered_backend_module import executorch_call_delegate
from executorch.exir.pass_base import PassResult


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


class _Concat(nn.Module):
    """Two inputs whose batch dims can differ, so one can derive from the other."""

    def forward(self, x, y):
        return torch.cat([x, y], dim=0)


def _derived_dim_ep():
    batch = torch.export.Dim("batch", min=2, max=32)
    return torch.export.export(
        _Concat().eval(),
        (torch.randn(4, 8), torch.randn(8, 8)),
        dynamic_shapes={"x": {0: batch}, "y": {0: 2 * batch}},
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


class _Independent(nn.Module):
    """Two inputs with unrelated dynamic dims, so one can be left un-enumerated."""

    def forward(self, x, y):
        return x.sum() + y.sum()


def _independent_dims_ep():
    batch = torch.export.Dim("batch", min=2, max=64)
    seq = torch.export.Dim("seq", min=2, max=64)
    return torch.export.export(
        _Independent().eval(),
        (torch.randn(4, 8), torch.randn(6, 8)),
        dynamic_shapes={"x": {0: batch}, "y": {0: seq}},
    )


class _DoubledIntermediate(nn.Module):
    """``cat`` doubles the dynamic dim, so a later boundary carries ``2*s``."""

    def __init__(self):
        super().__init__()
        self.first = nn.Linear(8, 8)
        self.second = nn.Linear(8, 8)

    def forward(self, x):
        h = torch.cat([x, x], 0)
        return self.second(self.first(h).relu()).relu()


def _doubled_intermediate_ep():
    batch = torch.export.Dim("batch", min=2, max=64)
    return torch.export.export(
        _DoubledIntermediate().eval(),
        (torch.randn(4, 8),),
        dynamic_shapes={"x": {0: batch}},
    )


def _addmm_nodes(graph_module):
    return [
        n
        for n in graph_module.graph.nodes
        if n.op == "call_function"
        and isinstance(n.target, EdgeOpOverload)
        and n.target._op.__name__ == "addmm.default"
    ]


class _TagSecondLinear:
    """Force a graph break so the delegate's inputs are intermediates."""

    def __call__(self, graph_module):
        addmms = _addmm_nodes(graph_module)
        if len(addmms) >= 2:
            second = addmms[1]
            do_not_delegate(second)
            # Its weight-transpose too, else a dangling single-node permute
            # delegate is left behind instead of a clean break.
            for inp in second.all_input_nodes:
                if isinstance(
                    inp.target, EdgeOpOverload
                ) and inp.target._op.__name__.endswith("_copy.default"):
                    do_not_delegate(inp)
        return PassResult(graph_module, True)


def _capture_shape_configs(exported_program, input_enumerations, transform_passes=None):
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
            transform_passes=transform_passes,
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

    def test_derived_dim_raises(self):
        """Substituting the symbol would pin 2*value, not the requested value."""
        with self.assertRaisesRegex(ValueError, "derived"):
            resolve_input_enumerations(_derived_dim_ep(), [None, {0: [8, 16]}])

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
        with self.assertRaisesRegex(Exception, "after substitution"):
            to_edge_transform_and_lower(
                _two_dynamic_dims_ep(),
                partitioner=[CoreAIPartitioner(input_enumerations=[{0: [2, 4]}])],
            ).to_executorch()


class BoundaryCoverageTest(unittest.TestCase):
    """Every symbolic dim at the boundary must be pinned by a specialization.

    ``set_static_shape_config`` treats a key as a whole-graph shape, so an input
    left symbolic in an entry is unconstrained there.
    """

    def test_unenumerated_sibling_input_is_rejected(self):
        """A second input's dynamic dim cannot be left out of the entries."""
        with self.assertRaisesRegex(Exception, "after substitution"):
            _capture_shape_configs(_independent_dims_ep(), [{0: [4, 16]}, None])

    def test_every_boundary_input_is_named(self):
        captured = _capture_shape_configs(_independent_dims_ep(), [{0: [4]}, {0: [6]}])
        self.assertTrue(captured, "set_static_shape_config was not called")
        for cfg in captured:
            for key, shapes in cfg.items():
                with self.subTest(key):
                    self.assertEqual(set(shapes), {"x", "y"})


class SubgraphBoundaryTest(unittest.TestCase):
    """The delegate boundary is a subgraph's, not necessarily the model's.

    Elsewhere here a single whole-graph partition makes the boundary inputs the
    model inputs, which hides the actual contract: an ET-input enumeration
    implies a *derived* enumeration at each subgraph, obtained by evaluating
    that boundary dim's own symbolic expression.
    """

    def _lower_split(self):
        """The two delegates' configs, as (model-input side, intermediate side).

        ``cat([x, x], 0)`` doubles the dim before the tagged linear, so the
        second delegate's boundary carries ``2*s``.
        """
        captured = _capture_shape_configs(
            _doubled_intermediate_ep(),
            [{0: [4, 16]}],
            transform_passes=[_TagSecondLinear()],
        )
        self.assertEqual(
            len(captured), 2, "expected a graph break into exactly 2 delegates"
        )
        # Identify by content rather than order, which preprocess does not fix.
        model_input = next(c for c in captured if "x" in next(iter(c.values())))
        intermediate = next(c for c in captured if c is not model_input)
        return model_input, intermediate

    def test_intermediate_boundary_enumerates_to_the_derived_values(self):
        model_input, intermediate = self._lower_split()
        self.assertEqual(
            sorted(tuple(s) for v in model_input.values() for s in v.values()),
            [(4, 8), (16, 8)],
        )
        # Declared [4, 16] on the ET input, but 2*s at this boundary.
        self.assertEqual(
            sorted(tuple(s) for v in intermediate.values() for s in v.values()),
            [(8, 8), (32, 8)],
        )

    def test_specialization_keys_align_across_delegates(self):
        """Keying on the symbol is what keeps the two sides paired.

        Delegate 0 produces (2s, 8) and delegate 1 consumes it, so under one key
        the two shapes have to correspond. Keying on local shape values instead
        would let them drift apart.
        """
        model_input, intermediate = self._lower_split()
        self.assertEqual(set(model_input), set(intermediate))
        for key, shapes in model_input.items():
            with self.subTest(key):
                (declared,) = shapes.values()
                (derived,) = intermediate[key].values()
                self.assertEqual(derived[0], 2 * declared[0])


if __name__ == "__main__":
    unittest.main()
