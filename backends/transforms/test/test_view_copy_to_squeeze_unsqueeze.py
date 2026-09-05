# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.transforms.view_copy_to_squeeze_unsqueeze import (
    ViewCopyToSqueezeUnsqueezePass,
)
from executorch.exir import to_edge


def _call_function_targets(graph_module: torch.fx.GraphModule):
    return [n.target for n in graph_module.graph.nodes if n.op == "call_function"]


class TestViewCopyToSqueezeUnsqueezePass(unittest.TestCase):
    def _run(self, model, example_inputs):
        aten = torch.export.export(model, example_inputs, strict=True)
        edge = to_edge(aten).transform([ViewCopyToSqueezeUnsqueezePass()])
        return _call_function_targets(edge.exported_program().graph_module)

    def test_unsqueeze_at_dim_0(self):
        # Adding a size-1 dim at the front (dim 0). The added dim index is 0,
        # which must not be treated as "no match" by a falsy check.
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.view(1, 3, 4)

        targets = self._run(Model().eval(), (torch.randn(3, 4),))
        unsqueeze_op = torch.ops.aten.unsqueeze_copy.default
        view_op = torch.ops.aten.view_copy.default
        self.assertTrue(
            any(t.name() == unsqueeze_op.name() for t in targets),
            f"view_copy was not converted to unsqueeze_copy: {targets}",
        )
        self.assertFalse(any(t.name() == view_op.name() for t in targets))

    def test_unsqueeze_at_dim_1(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.view(3, 1, 4)

        targets = self._run(Model().eval(), (torch.randn(3, 4),))
        unsqueeze_op = torch.ops.aten.unsqueeze_copy.default
        self.assertTrue(any(t.name() == unsqueeze_op.name() for t in targets))

    def test_squeeze(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.view(3, 4)

        targets = self._run(Model().eval(), (torch.randn(1, 3, 4),))
        squeeze_op = torch.ops.aten.squeeze_copy.dims
        self.assertTrue(any(t.name() == squeeze_op.name() for t in targets))


if __name__ == "__main__":
    unittest.main()
