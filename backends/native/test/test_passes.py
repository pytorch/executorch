# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from executorch.backends.native import get_default_compile_config
from executorch.backends.native.partitioner import NativePartitioner
from executorch.backends.native.passes import get_default_passes
from executorch.backends.native.passes.reinplace import NativeReinplacePass
from executorch.backends.native.serialization import deserialize_graph
from executorch.backends.native.test.utils import (
    _call_function_targets,
    _get_delegate_blob,
    _lower,
    _transformed,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.passes.cse_pass import CSEPass


class CSEPassTest(unittest.TestCase):
    def test_dedupes_identical_subexprs(self):
        class DupModel(nn.Module):
            def forward(self, x):
                a = x + x
                b = x + x
                return a * b

        ep = _transformed(DupModel(), (torch.randn(4, 4),), [CSEPass()])
        adds = [
            str(n.target)
            for n in ep.graph_module.graph.nodes
            if n.op == "call_function" and "add" in str(n.target)
        ]
        self.assertEqual(len(adds), 1, f"expected CSE to leave one add, got {adds}")


class NativeReinplacePassTest(unittest.TestCase):
    def test_rewrites_relu_in_place(self):
        class ReluModel(nn.Module):
            def forward(self, x):
                # relu on an intermediate (x + 1) can be rewritten in place;
                # relu directly on the immutable user input x cannot.
                return torch.relu(x + 1)

        ep = _transformed(ReluModel(), (torch.randn(4, 4),), [NativeReinplacePass()])
        targets = [
            str(n.target)
            for n in ep.graph_module.graph.nodes
            if n.op == "call_function"
        ]
        self.assertTrue(
            any("relu_" in t for t in targets),
            f"expected in-place relu_, got {targets}",
        )


class ReplaceCopyWithAliasPassTest(unittest.TestCase):
    def test_alias_ops_serialize_valid_targets(self):
        # ReplaceCopyWithAliasPass rewrites *_copy view ops to plain aten
        # OpOverloads (e.g. transpose_copy -> transpose). Those must serialize to
        # real op names, not the bare "torch._ops.aten." from over-unwrapping.
        class ViewModel(nn.Module):
            def forward(self, x):
                return x.transpose(0, 1).reshape(-1) + 1.0

        blob = _get_delegate_blob(_lower(ViewModel(), (torch.randn(3, 4),)))
        graph = deserialize_graph(blob)
        targets = [t for t in _call_function_targets(graph) if t]
        # The pass rewrites at least one *_copy view op to its aliasing form.
        alias_names = ("transpose", "permute", "view", "slice", "unsqueeze", "expand")
        self.assertTrue(
            any(
                any(f"aten.{a}." in t for a in alias_names) and "_copy." not in t
                for t in targets
            ),
            f"expected an aliasing view op, got {targets}",
        )
        for t in targets:
            self.assertFalse(
                t.startswith("torch._ops.") or t.endswith("."),
                f"malformed serialized target: {t!r}",
            )

    def test_dynamic_view_converts_to_alias(self):
        # A view with a symbolic size (dynamic dim) on a contiguous input must be
        # rewritten to a true aten.view, not conservatively left as view_copy.
        class DynView(nn.Module):
            def forward(self, x):
                return x.reshape(x.shape[0], -1) + 1.0

        ep = torch.export.export(
            DynView(),
            (torch.randn(4, 2, 3),),
            dynamic_shapes={"x": {0: torch.export.Dim("b", max=1024)}},
        )
        edge = to_edge_transform_and_lower(
            ep,
            transform_passes=get_default_passes(),
            partitioner=[NativePartitioner()],
            compile_config=get_default_compile_config(),
        )
        graph = deserialize_graph(_get_delegate_blob(edge))
        targets = _call_function_targets(graph)
        self.assertIn("torch.ops.aten.view.default", targets)
        self.assertNotIn("torch.ops.aten.view_copy.default", targets)
