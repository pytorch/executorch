# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from executorch.backends.native import get_default_compile_config
from executorch.exir import to_edge
from executorch.exir.passes.cse_pass import CSEPass


def _transformed(model, example_inputs, passes):
    edge = to_edge(
        torch.export.export(model, example_inputs),
        compile_config=get_default_compile_config(),
    )
    return edge.transform(passes).exported_program()


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
