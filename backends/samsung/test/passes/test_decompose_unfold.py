# Copyright (c) 2026 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.samsung._passes.decompose_unfold import DecomposeUnfold


class Unfold(torch.nn.Module):
    def __init__(self, dim: int, size: int, step: int) -> None:
        super().__init__()
        self.dim = dim
        self.size = size
        self.step = step

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unfold(self.dim, self.size, self.step)


class TestDecomposeUnfold(unittest.TestCase):
    def test_matches_unfold(self):
        # (input shape, dim, size, step); size == N * step takes the reshape
        # path, the others slice each window, including single windows.
        cases = [
            ((2, 10, 3), 1, 4, 2),
            ((1, 8, 6), 2, 2, 2),
            ((2, 10, 3), 1, 3, 2),
            ((2, 6, 3), 1, 2, 4),
            ((2, 3, 4), 1, 3, 2),
            ((2, 2, 3), 1, 1, 2),
        ]
        for shape, dim, size, step in cases:
            with self.subTest(shape=shape, dim=dim, size=size, step=step):
                module = Unfold(dim, size, step)
                x = torch.randn(*shape)
                graph_module = torch.export.export(module, (x,)).module()
                DecomposeUnfold()(graph_module)
                self.assertFalse(
                    any(
                        node.target == torch.ops.aten.unfold.default
                        for node in graph_module.graph.nodes
                    )
                )
                self.assertTrue(torch.equal(graph_module(x), module(x)))


if __name__ == "__main__":
    unittest.main()
