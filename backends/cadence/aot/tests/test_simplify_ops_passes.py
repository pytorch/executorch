# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest
from typing import cast, Optional, Tuple

import executorch.backends.cadence.aot.ops_registrations  # noqa
import torch
from executorch.backends.cadence.aot.graph_builder import single_op_builder
from executorch.backends.cadence.aot.pass_utils import count_node
from executorch.backends.cadence.aot.simplify_ops import (
    BindOptionalArgsPass,
    SimplifySliceOpPass,
)
from executorch.backends.cadence.aot.typing_stubs import expand
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.passes.infra.pass_base import PassResult


class TestSimplifyOpsPasses(unittest.TestCase):
    @expand(
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
    ) -> None:
        x = torch.randn(*in_shape)
        y = torch.randn(*src_shape)
        gm = single_op_builder(
            placeholders=(x, y),
            op=exir_ops.edge.aten.slice_scatter.default,
            args=(x, y, dim, start, end, step),
        )
        p = SimplifySliceOpPass()
        gm = cast(PassResult, p(gm)).graph_module
        self.assertEqual(count_node(gm, exir_ops.edge.aten.slice_scatter.default), 0)

    @expand(
        [
            [(3, 16, 5), 1, 15, 3, 3],
        ]
    )
    @torch.no_grad()
    def test_simplify_slice_op(
        self,
        in_shape: Tuple[int],
        dim: int,
        start: Optional[int] = None,
        end: Optional[int] = None,
        step: int = 1,
    ) -> None:
        x = torch.randn(*in_shape)
        gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.slice_copy.Tensor,
            args=(
                x,
                dim,
                start,
                end,
                step,
            ),
        )
        p = SimplifySliceOpPass()
        gm = cast(PassResult, p(gm)).graph_module
        self.assertEqual(count_node(gm, exir_ops.edge.aten.slice_copy.Tensor), 0)
        self.assertEqual(count_node(gm, exir_ops.edge.aten.full.default), 1)

    def test_simplify_slice_op_args(self) -> None:
        x = torch.rand(4, 5)
        gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.slice_copy.Tensor,
            args=(x, 1),
            kwargs={"end": 3},
        )
        original_slice_copy = list(gm.graph.nodes)[1]
        self.assertEqual(original_slice_copy.args[1:], (1,))
        self.assertEqual(original_slice_copy.kwargs, {"end": 3})
        gm = BindOptionalArgsPass().call(gm).graph_module
        modified_slice_copy = list(gm.graph.nodes)[1]
        self.assertEqual(modified_slice_copy.args[1:], (1, None, 3, 1))
        self.assertEqual(modified_slice_copy.kwargs, {})
