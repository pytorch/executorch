import unittest

import torch
from executorch.backends.transforms.rank_0_to_rank_1 import Rank0ToRank1Pass
from executorch.exir import to_edge


class TestRank0ToRank1Pass(unittest.TestCase):
    def test_pass(
        self,
    ):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        model = Model()
        model.eval()

        example_inputs = (torch.tensor(1.0), torch.tensor(2.0))
        aten = torch.export.export(model, example_inputs)

        # Check that the input rank is 0
        for node in aten.graph.nodes:
            if node.op == "placeholder":
                self.assertTrue(node.meta["val"].shape == ())

        edge = to_edge(aten).transform([Rank0ToRank1Pass()])

        # Check that the input rank is 1
        for node in edge.exported_program().graph.nodes:
            if node.op == "placeholder":
                self.assertTrue(node.meta["val"].shape == (1, 1))
