# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import unittest

import executorch.backends.cadence.aot.ops_registrations  # noqa
import torch
from executorch.backends.cadence.aot.compiler import (
    export_to_edge,
    quantize_and_export_to_cadence,
)
from executorch.backends.cadence.aot.fuse_ops import FuseQuantDequantToRequantizePass
from executorch.backends.cadence.aot.pass_utils import (
    count_node,
    get_compute_nodes_in_gm,
    nodes_not_adjacent_in_gm,
    nodes_not_connected_in_gm,
)
from executorch.backends.cadence.aot.reorder_ops import (
    AdvanceQuantizeOpAboveDefInBranchPass,
    PostponeDequantizeOpBelowUseChainPass,
    PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView,
)
from executorch.exir.dialects._ops import ops as exir_ops


class TestReorderPasses(unittest.TestCase):
    def test_sink_dequantize(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(6, 12, bias=False)

            def forward(self, x, y):
                x1 = self.linear(x)
                y1 = self.linear(y)
                x2 = torch.ops.aten.abs(x1)
                return torch.ops.aten.cat((x2, y1))

        inputs = (torch.randn(32, 6), torch.randn(32, 6))
        graph_module = (
            quantize_and_export_to_cadence(M(), inputs).exported_program().graph_module
        )
        # Expect the SinkDequant pass to move dequant(y) from above the relu to just below it
        self.assertTrue(
            nodes_not_adjacent_in_gm(
                graph_module,
                exir_ops.edge.aten.abs.default,
                exir_ops.edge.aten.cat.default,
            ),
        )
        self.assertTrue(
            nodes_not_adjacent_in_gm(
                graph_module,
                exir_ops.edge.cadence.dequantize_per_tensor.default,
                exir_ops.edge.cadence.dequantize_per_tensor.default,
            ),
        )

    def test_advance_branched_quantize(self):
        class ReorderOpsBranch(torch.nn.Module):
            def forward(self, x):
                x = x.view((32, 6))
                x1 = torch.slice_copy(x, dim=0, start=0, end=6, step=1)
                x1 = exir_ops.edge.quantized_decomposed.quantize_per_tensor(
                    x1, 0.1, 10, 0, 255, torch.uint8
                )
                x2 = torch.slice_copy(x, dim=0, start=6, end=12, step=1)
                x2 = exir_ops.edge.quantized_decomposed.quantize_per_tensor(
                    x2, 0.1, 10, 0, 255, torch.uint8
                )
                x3 = torch.slice_copy(x, dim=0, start=12, end=18, step=1)
                x3 = exir_ops.edge.quantized_decomposed.quantize_per_tensor(
                    x3, 0.1, 10, 0, 255, torch.uint8
                )
                x4 = torch.slice_copy(x, dim=0, start=18, end=24, step=1)
                x4 = exir_ops.edge.quantized_decomposed.quantize_per_tensor(
                    x4, 0.2, 4, 0, 255, torch.uint8
                )
                return (x1, x2, x3, x4)

        model = ReorderOpsBranch()
        X = torch.randn(64, 3)
        graph_module = export_to_edge(model, (X,)).exported_program().graph_module
        graph_module = AdvanceQuantizeOpAboveDefInBranchPass()(
            graph_module
        ).graph_module
        graph_module.graph.eliminate_dead_code()
        nodes = get_compute_nodes_in_gm(graph_module)
        # The quantize op should be hoisted to dominate the branch
        self.assertTrue(
            nodes[0] == exir_ops.edge.quantized_decomposed.quantize_per_tensor
        )
        # There should be 5 quantize ops: the 4 originally present in the model,
        # and the one that was hoisted above the slices
        self.assertEqual(
            count_node(
                graph_module,
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            ),
            5,
        )
        # Ensure none of the slice nodes were erroneously removed
        self.assertEqual(
            count_node(
                graph_module,
                exir_ops.edge.aten.slice_copy.Tensor,
            ),
            4,
        )
        # Each of the 4 original quant ops should now be paired with a dequant op
        self.assertEqual(
            count_node(
                graph_module,
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            ),
            4,
        )
        graph_module = FuseQuantDequantToRequantizePass()(graph_module).graph_module
        # We expect 3 dequant/quant pairs to be removed because they have matching params,
        # leaving a single dequant/quant pair that is then merged into a requantize op
        self.assertEqual(
            count_node(
                graph_module,
                exir_ops.edge.cadence.requantize.default,
            ),
            1,
        )

    @torch.no_grad()
    def test_advance_quantize(self):
        class ReorderOpsChain(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(6, 12, bias=False)

            def forward(self, x):
                x = x.permute([1, 0, 3, 2])
                x = self.linear(x)
                return x

        model = ReorderOpsChain()
        X = torch.randn(16, 1, 6, 32)

        graph_module = (
            quantize_and_export_to_cadence(model, (X,)).exported_program().graph_module
        )
        # Assert that the quant node is no longer the successor of
        # permute node.
        self.assertTrue(
            nodes_not_connected_in_gm(
                graph_module,
                exir_ops.edge.aten.permute_copy.default,
                exir_ops.edge.cadence.quantize_per_tensor.default,
            ),
        )
        # Assert that permute node is the successor of quant node
        self.assertFalse(
            nodes_not_connected_in_gm(
                graph_module,
                exir_ops.edge.cadence.quantize_per_tensor.default,
                exir_ops.edge.aten.permute_copy.default,
            ),
        )

    def test_postpone_dequantize(self):
        class ReorderOpsChain(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(6, 12, bias=False)

            def forward(self, x):
                x = self.linear(x)
                x = x.permute([1, 0, 3, 2])
                return x

        model = ReorderOpsChain()
        X = torch.randn(1, 16, 32, 6)

        graph_module = (
            quantize_and_export_to_cadence(model, (X,)).exported_program().graph_module
        )
        # Assert that the dequant node is no longer the predecessor of the permute node
        self.assertTrue(
            nodes_not_connected_in_gm(
                graph_module,
                exir_ops.edge.cadence.dequantize_per_tensor.default,
                exir_ops.edge.aten.permute_copy.default,
            ),
        )
        # Assert that dequant node is the successor of permute node
        self.assertFalse(
            nodes_not_connected_in_gm(
                graph_module,
                exir_ops.edge.aten.permute_copy.default,
                exir_ops.edge.cadence.dequantize_per_tensor.default,
            ),
        )

    def test_postpone_dequantize_branched(self):
        class ReorderOpsBranch(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 12, bias=False)

            def forward(self, x):
                x0 = exir_ops.edge.quantized_decomposed.dequantize_per_tensor(
                    x, 0.1, 10, 0, 255, torch.uint8
                )
                x0 = torch.squeeze(x0, 0)
                x1 = torch.slice_copy(x0, dim=0, start=0, end=6, step=1)
                x1 = self.linear(x1)

                x2 = torch.slice_copy(x0, dim=0, start=6, end=12, step=1)
                x2 = self.linear(x2)

                x3 = torch.slice_copy(x0, dim=0, start=12, end=18, step=1)
                x3 = self.linear(x3)

                return (x1, x2, x3)

        model = ReorderOpsBranch()
        X = torch.randint(0, 255, [1, 18, 3], dtype=torch.uint8)
        graph_module = export_to_edge(model, (X,)).exported_program().graph_module
        graph_module = PostponeDequantizeOpBelowUseChainPass()(
            graph_module
        ).graph_module
        graph_module.graph.eliminate_dead_code()

        # Asset that the dequant node was split into 4, one per branch
        self.assertEqual(
            count_node(
                graph_module,
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            ),
            3,
        )

        # Assert that the dequant node is no longer the predecessor of the squeeze node
        self.assertTrue(
            nodes_not_connected_in_gm(
                graph_module,
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
                exir_ops.edge.aten.squeeze_copy.dims,
            ),
        )
        # Assert that dequant node is not predecessor of slice (it should've been moved below slice)
        self.assertTrue(
            nodes_not_connected_in_gm(
                graph_module,
                exir_ops.edge.cadence.dequantize_per_tensor.default,
                exir_ops.edge.aten.slice_copy.Tensor,
            ),
        )

    # 4d -> permute -> 4d -> view -> 3d
    def test_permute3_view4_chains(self):
        class PermuteViewChain(torch.nn.Module):
            def forward(self, x):
                # x is [3, 1, 768]
                x = x.view((3, 12, 64))
                # x is [3, 12, 64]
                x = x.permute([1, 0, 2])
                # x is [12, 3, 64]
                x = x.view((1, 12, 3, 64))
                # x is [1, 12, 3, 64]
                x = x.permute([0, 1, 3, 2])
                # x is [1, 12, 64, 3]
                return x

        model = PermuteViewChain()
        X = torch.randn(3, 1, 768)
        graph_module = export_to_edge(model, (X,)).exported_program().graph_module

        # Performing transform
        graph_module = PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView()(
            graph_module
        ).graph_module
        graph_module.graph.eliminate_dead_code()

        # Assert the order becomes view, view, permute, permute
        nodes = get_compute_nodes_in_gm(graph_module)
        self.assertEqual(len(nodes), 4)
        self.assertTrue(nodes[0] == exir_ops.edge.aten.view_copy)
        self.assertTrue(nodes[1] == exir_ops.edge.aten.view_copy)
        self.assertTrue(nodes[2] == exir_ops.edge.aten.permute_copy)
        self.assertTrue(nodes[3] == exir_ops.edge.aten.permute_copy)

    # 3d -> permute -> 3d -> view -> 4d
    def test_permute4_view3_chains(self):
        class PermuteViewChain(torch.nn.Module):
            def forward(self, x):
                # x is [3, 1, 768]
                x = x.view((1, 3, 12, 64))
                # x is [1, 3, 12, 64]
                x = x.permute([3, 1, 0, 2])
                # x is [64, 3, 1, 12]
                x = x.view((64, 3, 12))
                # x is [64, 3, 12]
                x = x.permute([2, 1, 0])
                # x is [12, 3, 64]
                return x

        model = PermuteViewChain()
        X = torch.randn(3, 1, 768)
        graph_module = export_to_edge(model, (X,)).exported_program().graph_module

        # Performing transform
        graph_module = PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView()(
            graph_module
        ).graph_module
        graph_module.graph.eliminate_dead_code()

        # Assert the order becomes view, view, permute, permute
        nodes = get_compute_nodes_in_gm(graph_module)
        self.assertEqual(len(nodes), 4)
        self.assertTrue(nodes[0] == exir_ops.edge.aten.view_copy)
        self.assertTrue(nodes[1] == exir_ops.edge.aten.view_copy)
        self.assertTrue(nodes[2] == exir_ops.edge.aten.permute_copy)
        self.assertTrue(nodes[3] == exir_ops.edge.aten.permute_copy)

    # Negative test case where the transform should not happen.
    # permute->4d->view->3d where the view not only removes the dimension whose
    # size is 1 (this is ok), but also changes the size of the dimensions (not ok).
    def test_permute_view_chains_neg(self):
        class PermuteViewChain(torch.nn.Module):
            def forward(self, x):
                # x is [3, 1, 768]
                x = x.view((1, 3, 12, 64))
                # x is [1, 3, 12, 64]
                x = x.permute([3, 1, 0, 2])
                # x is [64, 3, 1, 12]
                x = x.view((64, 6, 6))
                # x is [64, 6, 6]
                x = x.permute([2, 1, 0])
                # x is [6, 6, 64]
                return x

        model = PermuteViewChain()
        X = torch.randn(3, 1, 768)
        graph_module = export_to_edge(model, (X,)).exported_program().graph_module

        # Performing transform (nothing should happen)
        graph_module = PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView()(
            graph_module
        ).graph_module
        graph_module.graph.eliminate_dead_code()

        # Assert the order is still view, permute, view, permute
        nodes = get_compute_nodes_in_gm(graph_module)
        self.assertEqual(len(nodes), 4)
        self.assertTrue(nodes[0] == exir_ops.edge.aten.view_copy)
        self.assertTrue(nodes[1] == exir_ops.edge.aten.permute_copy)
        self.assertTrue(nodes[2] == exir_ops.edge.aten.view_copy)
        self.assertTrue(nodes[3] == exir_ops.edge.aten.permute_copy)
