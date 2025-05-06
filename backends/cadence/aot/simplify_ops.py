# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


# This file contains all the functions that simplify args of an op

import sys
from typing import Optional

from executorch.backends.cadence.aot.pass_utils import (
    CadencePassAttribute,
    register_cadence_pass,
)
from executorch.exir import ExportedProgram

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, ProxyValue


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class SimplifySliceOpPass(ExportPass):
    """
    Simplify the start and end indices of slice and slice_scatter ops.
    """

    def adjust_slice_range(
        self,
        length: int,
        start: Optional[int] = None,
        end: Optional[int] = None,
        step: int = 1,
    ) -> tuple[int, int]:
        # Get the start index and end index
        start_val = start if start is not None else 0
        end_val = end if end is not None else sys.maxsize  # 2^63 – 1

        # If start_val and end_val are negative, add length to them
        if start_val < 0:
            start_val += length
        if end_val < 0:
            end_val += length

        # If the start val is still outside the tensor_size along the sliced
        # dimension, adjust it accordingly.
        if start_val < 0:
            start_val = 0
        elif start_val >= length:
            start_val = length

        # If the end val is still outside the tensor_size along the sliced
        # dimension, adjust it accordingly.
        if end_val < start_val:
            end_val = start_val
        elif end_val >= length:
            end_val = length

        # Return the adjusted start and end indices
        return (start_val, end_val)

    def call_operator(self, op, args, kwargs, meta):
        # We are only interested in slice_copy or slice_scatter ops
        if op not in {
            exir_ops.edge.aten.slice_copy.Tensor,
            exir_ops.edge.aten.slice_scatter.default,
        }:
            return super().call_operator(op, args, kwargs, meta)

        # Check if it is a slice_scatter op or not. The slice_scatter op has
        # an extra src argument at index 1.
        slice_scatter = op == exir_ops.edge.aten.slice_scatter.default
        # Parse the arguments
        # Extract the tensor to be sliced, and the slicing dimension
        in_tensor = args[0].to_tensor() if isinstance(args[0], ProxyValue) else args[0]
        dim = args[1 + slice_scatter] if len(args) > 1 + slice_scatter else 0
        # Make dim non-negative
        dim = dim if dim >= 0 else dim + in_tensor.dim()
        length = in_tensor.size(dim)

        # Get the adjusted start and end indices
        start_val = args[2 + slice_scatter] if len(args) > 2 + slice_scatter else None
        end_val = args[3 + slice_scatter] if len(args) > 3 + slice_scatter else None
        step = args[4 + slice_scatter] if len(args) > 4 + slice_scatter else 1
        (start_val, end_val) = self.adjust_slice_range(length, start_val, end_val, step)

        # If the start_val is geq end_val, then we can return an empty tensor
        # for slice op, or input for slice_scatter op.
        if start_val >= end_val and slice_scatter:
            return args[0]
        if start_val >= end_val:
            empty_shape = [x for x in in_tensor.shape if x != 0]
            empty_shape[dim] = 0
            return super().call_operator(
                exir_ops.edge.aten.full.default,
                (tuple(empty_shape), 0),
                {"dtype": in_tensor.dtype},
                meta,
            )

        # Create new args
        new_args = (
            (args[0],)
            + ((args[1],) if slice_scatter else ())
            + (dim, start_val, end_val, step)
        )
        return super().call_operator(op, new_args, kwargs, meta)


def FuseMulToQuantPass(
    exported_program: ExportedProgram,
) -> ExportedProgram:
    """
    If a mul op using a scalar constant input is followed by a quantize op, we can fuse the mul
    into the quantize op by updating its scale. Unfortunately, lifted constants are not stored in
    the nodes, so we need to find the constant value in the constants dict.
    """
    graph = exported_program.graph_module.graph
    for node in graph.nodes:
        # We are only interested in mul ops
        if node.target != exir_ops.edge.aten.mul.Tensor:
            continue

        # Only applies if the following op is a quantize op
        user = list(node.users.keys())[0]
        if user.target not in (
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            exir_ops.edge.cadence.quantize_per_tensor.default,
        ):
            continue

        # Check that the second arg of the mul is a constant
        # Get the constant value
        if node.args[1].name not in exported_program.state_dict:
            continue

        tensor = exported_program.state_dict[node.args[1].name]

        args = list(user.args)

        # Update the scale of the quantize op
        args[0] = node.args[0]
        args[1] = user.args[1] / tensor.item()

        # Return the op with the updated args
        with graph.inserting_before(user):
            op_node = graph.call_function(user.target, args=tuple(args))
            op_node.meta = node.meta
        user.replace_all_uses_with(op_node)

    exported_program.graph_module.recompile()
    exported_program.graph_module.graph.eliminate_dead_code()

    return exported_program


# This class encapsulates all the functions that simplify the op's args
class CadenceSimplifyOpsInGraph:
    passes = [
        SimplifySliceOpPass,
    ]
