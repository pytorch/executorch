import unittest

import torch
from executorch.backends.qualcomm._passes import InsertReshapeForReduceOps, ConvertPadToSliceConcat


class TestPasses(unittest.TestCase):
    def test_insert_reshape_for_reduced_ops(self):
        class ArgmaxModule(torch.nn.Module):
            def forward(self, x):
                return torch.argmax(x, dim=None)

        mod = ArgmaxModule()

        x = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
        ep = torch.export.export(mod, (x,))
        # Run original module for reference
        ref = mod(x)

        reshape_nodes = [
            n for n in ep.graph.nodes if n.target == torch.ops.aten.reshape.default
        ]
        argmax_nodes = [
            n for n in ep.graph.nodes if n.target == torch.ops.aten.argmax.default
        ]
        self.assertTrue(len(reshape_nodes) == 0, "Reshape node not inserted")
        self.assertTrue(len(argmax_nodes) == 1, "Argmax node missing")

        InsertReshapeForReduceOps()(ep.graph_module)

        out = ep.graph_module(x)

        # Check graph structure: argmax should take a reshape as input
        reshape_nodes = [
            n for n in ep.graph.nodes if n.target == torch.ops.aten.reshape.default
        ]
        argmax_nodes = [
            n for n in ep.graph.nodes if n.target == torch.ops.aten.argmax.default
        ]
        self.assertTrue(len(reshape_nodes) == 1, "Reshape node should be inserted")
        self.assertTrue(len(argmax_nodes) == 1, "Argmax node missing")

        argmax_node = argmax_nodes[0]
        self.assertEqual(argmax_node.args[1], 0, "Argmax dim not set to 0")

        # Execute new graph and compare with reference
        out = ep.graph_module(x)
        self.assertTrue(
            torch.equal(*out, ref), f"Output mismatch: got {out}, expected {ref}"
        )

    def test_convert_pad_to_slice_concat(self):
        # Test with circular and replicate modes, the pass should remove the pad node and insert slice and concat nodes
        class Pad(torch.nn.Module):
            def __init__(self, mode):
                super().__init__()
                self.mode = mode

            def forward(self, x):
                # pad order = [left, right, top, bottom]
                return torch.ops.aten.pad.default(x, (1, 1, 1, 1), mode=self.mode)

        modes = ["circular", "replicate"]
        for mode in modes:
            mod = Pad(mode)
            x = torch.arange(1.0, 17.0).reshape(1, 1, 4, 4)
            ep = torch.export.export(mod, (x,))
            # Run original module for reference
            ref = mod(x)

            circular_pad_nodes = [
                n for n in ep.graph.nodes if n.target == torch.ops.aten.pad.default and mode in n.args
            ]
            self.assertTrue(len(circular_pad_nodes) == 1, "Circular pad node missing")

            ConvertPadToSliceConcat()(ep.graph_module)

            out = ep.graph_module(x)
            # Check graph structure: argmax should take a reshape as input
            slice_nodes = [
                n for n in ep.graph.nodes if n.target == torch.ops.aten.slice.Tensor
            ]
            circular_pad_nodes = [
                n for n in ep.graph.nodes if n.target == torch.ops.aten.pad.default and mode in n.args
            ]
            self.assertTrue(len(slice_nodes) >= 1, "Slice node should be inserted")
            self.assertTrue(len(circular_pad_nodes) == 0, "Pad node should be removed")

            # Execute new graph and compare with reference
            out = ep.graph_module(x)
            self.assertTrue(
                torch.equal(*out, ref), f"Output mismatch: got {out}, expected {ref}"
            )


if __name__ == "__main__":
    unittest.main()
