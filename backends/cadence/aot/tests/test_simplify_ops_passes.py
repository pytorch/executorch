# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
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
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.utils import _pytree as pytree


def transform_and_check_numerics(
    original_graph: torch.fx.GraphModule,
    inputs: tuple[torch.Tensor, ...] | list[torch.Tensor],
    pass_to_run: PassBase,
    pass_name: str,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> PassResult:
    """Run a graph transformation and validate numerical equivalence.

    Args:
        original_graph: The original graph module before transformation
        inputs: Input tensors to run through both graphs
        pass_to_run: The pass to apply to the graph
        pass_name: Name of the pass being validated (for error messages)
        rtol: Relative tolerance for allclose comparison
        atol: Absolute tolerance for allclose comparison

    Returns:
        The PassResult from the transformation
    """
    # Deepcopy to preserve original for comparison
    gm_before = copy.deepcopy(original_graph)

    # Run the transformation
    result = cast(PassResult, pass_to_run.call(original_graph))

    # Validate numerical equivalence
    gm_before.eval()
    result.graph_module.eval()
    with torch.no_grad():
        orig_out = gm_before(*inputs)
        mod_out = result.graph_module(*inputs)

    flat_orig_out, _ = pytree.tree_flatten(orig_out)
    flat_mod_out, _ = pytree.tree_flatten(mod_out)

    # Check that outputs match within tolerance
    for i, (orig_tensor, mod_tensor) in enumerate(zip(flat_orig_out, flat_mod_out)):
        if not torch.allclose(orig_tensor, mod_tensor, rtol=rtol, atol=atol):
            max_diff = torch.max(torch.abs(orig_tensor - mod_tensor)).item()
            raise AssertionError(
                f"Pass validation failed for pass {pass_name}. "
                f"Output tensor {i} differs by max {max_diff:.6e}. "
                f"Expected rtol={rtol}, atol={atol}. "
                f"Original output: {orig_tensor}, Modified output: {mod_tensor}"
            )

    return result


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
        result = transform_and_check_numerics(
            gm, (x, y), SimplifySliceOpPass(), "SimplifySliceOpPass"
        )
        self.assertTrue(result.modified)
        gm = result.graph_module
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
        result = transform_and_check_numerics(
            gm, (x,), SimplifySliceOpPass(), "SimplifySliceOpPass"
        )
        self.assertTrue(result.modified)
        gm = result.graph_module
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
        result = transform_and_check_numerics(
            gm, (x,), BindOptionalArgsPass(), "BindOptionalArgsPass"
        )
        self.assertTrue(result.modified)
        gm = result.graph_module
        modified_slice_copy = list(gm.graph.nodes)[1]
        self.assertEqual(modified_slice_copy.args[1:], (1, None, 3, 1))
        self.assertEqual(modified_slice_copy.kwargs, {})
