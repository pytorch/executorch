# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import unittest
from typing import cast, Optional, Tuple

import executorch.backends.cadence.aot.ops_registrations  # noqa
import torch
from executorch.backends.cadence.aot.compiler import export_to_edge
from executorch.backends.cadence.aot.pass_utils import count_node
from executorch.backends.cadence.aot.simplify_ops import SimplifySliceOpPass
from executorch.exir.dialects._ops import ops as exir_ops
from parameterized.parameterized import parameterized
from torch.fx.passes.infra.pass_base import PassResult


class TestSimplifyOpsPasses(unittest.TestCase):
    @parameterized.expand(
        [
            [(3, 16, 5), (3, 0, 5), 1, 15, 3, 3],
        ]
    )
    @torch.no_grad()
    def test_simplify_slice_scatter_op(
        self,
        in_shape: Tuple[int],
        src_shape: Tuple[int],
        dim: int,
        start: Optional[int] = None,
        end: Optional[int] = None,
        step: int = 1,
    ):
        class SliceScatter(torch.nn.Module):
            def __init__(
                self, dim: int, start: Optional[int], end: Optional[int], step: int
            ):
                super().__init__()
                self.dim = dim
                self.start = start
                self.end = end
                self.step = step

            def forward(self, x: torch.Tensor, y: torch.Tensor):
                return torch.slice_scatter(
                    x, y, self.dim, self.start, self.end, self.step
                )

        model = SliceScatter(dim, start, end, step)
        x = torch.randn(in_shape)
        y = torch.randn(src_shape)
        graph_module = export_to_edge(model, (x, y)).exported_program().graph_module

        p = SimplifySliceOpPass()

        graph_after_passes = cast(PassResult, p(graph_module)).graph_module

        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.slice_scatter.default), 0
        )

    @parameterized.expand(
        [
            [(3, 16, 5), (3, 0, 5), 1, 15, 3, 3],
        ]
    )
    @torch.no_grad()
    def test_simplify_slice_op(
        self,
        in_shape: Tuple[int],
        src_shape: Tuple[int],
        dim: int,
        start: Optional[int] = None,
        end: Optional[int] = None,
        step: int = 1,
    ):
        class SliceCopy(torch.nn.Module):
            def __init__(
                self, dim: int, start: Optional[int], end: Optional[int], step: int
            ):
                super().__init__()
                self.dim = dim
                self.start = start
                self.end = end
                self.step = step

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.slice_copy(
                    x, dim=self.dim, start=self.start, end=self.end, step=self.step
                )

        # Create a model with single slice copy op.
        model = SliceCopy(dim, start, end, step)
        x = torch.randn(in_shape)
        graph_module = export_to_edge(model, (x,)).exported_program().graph_module
        self.assertEqual(
            count_node(graph_module, exir_ops.edge.aten.slice_copy.Tensor), 1
        )

        p = SimplifySliceOpPass()

        graph_after_passes = cast(PassResult, p(graph_module)).graph_module

        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.slice_copy.Tensor), 0
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.full.default), 1
        )
