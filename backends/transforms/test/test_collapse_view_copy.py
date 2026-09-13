# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from executorch.backends.transforms.collapse_view_copy import CollapseViewCopyPass
from executorch.exir.dialects._ops import ops as _edge_ops


class CollapseViewCopyPassTest(unittest.TestCase):
    def test_collapses_chain(self):
        # view_copy(view_copy(x)) collapses to a single view_copy on x.
        vc = _edge_ops.edge.aten.view_copy.default
        g = torch.fx.Graph()
        x = g.placeholder("x")
        x.meta["val"] = torch.zeros(6)
        v1 = g.call_function(vc, (x, [2, 3]))
        v1.meta["val"] = torch.zeros(2, 3)
        v2 = g.call_function(vc, (v1, [3, 2]))
        v2.meta["val"] = torch.zeros(3, 2)
        g.output((v2,))
        gm = torch.fx.GraphModule(nn.Module(), g)

        result = CollapseViewCopyPass().call(gm)
        self.assertTrue(result.modified)
        view_copies = [
            n for n in gm.graph.nodes if n.op == "call_function" and n.target is vc
        ]
        self.assertEqual(len(view_copies), 1)
        self.assertIs(view_copies[0].args[0], x)
