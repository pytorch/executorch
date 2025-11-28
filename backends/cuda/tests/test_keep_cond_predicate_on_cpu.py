import unittest

import torch
from executorch.backends.cuda.passes.keep_cond_predicate_on_cpu import (
    KeepCondPredicateOnCpuPass,
)
from torch.export import export


class TestKeepCondPredicateOnCpuPass(unittest.TestCase):
    def test_keep_cond_predicate_on_cpu(self):
        # Define a simple model using torch.cond
        class Model(torch.nn.Module):
            def forward(self, pred, x, y):
                def true_fn(x, y):
                    return x + y

                def false_fn(x, y):
                    return x - y

                return torch.cond(pred, true_fn, false_fn, [x, y])

        model = Model()
        pred = torch.tensor(True)
        x = torch.randn(2, 2)
        y = torch.randn(2, 2)

        # Export the model
        ep = export(model, (pred, x, y))
        gm = ep.graph_module

        # Simulate move_to_device_pass by setting all placeholders to cuda using FakeTensorMode
        # We need to be careful not to trigger CUDA init
        from unittest.mock import MagicMock

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                if "val" in node.meta:
                    # Use MagicMock to simulate a tensor on cuda
                    val = MagicMock(spec=torch.Tensor)
                    val.device = torch.device("cuda")

                    def to_side_effect(device):
                        new_val = MagicMock(spec=torch.Tensor)
                        new_val.device = torch.device(device)
                        return new_val

                    val.to.side_effect = to_side_effect
                    node.meta["val"] = val

        # Verify that pred is on cuda
        pred_node = list(gm.graph.nodes)[0]
        self.assertEqual(pred_node.meta["val"].device.type, "cuda")

        # Run the pass
        pass_instance = KeepCondPredicateOnCpuPass()
        pass_instance(gm)

        # Verify that pred is back on cpu
        self.assertEqual(pred_node.meta["val"].device.type, "cpu")

        # Verify other nodes are still on cuda (if they were)
        # The second node is x
        x_node = list(gm.graph.nodes)[1]
        self.assertEqual(x_node.meta["val"].device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
