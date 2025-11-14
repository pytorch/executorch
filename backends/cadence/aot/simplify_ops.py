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
from executorch.backends.cadence.aot.utils import rebind
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass


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
        in_tensor = args[0].to_tensor()
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


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class BindOptionalArgsPass(ExportPass):
    """Bind all optional args and kwargs."""

    def call_operator(self, op, args, kwargs, meta):
        if not isinstance(op, EdgeOpOverload):
            return super().call_operator(op, args, kwargs, meta)

        if (updated_args := rebind(op, args, kwargs)) is not None:
            args, kwargs = updated_args

        return super().call_operator(op, args, kwargs, meta)


# This class encapsulates all the functions that simplify the op's args
class CadenceSimplifyOpsInGraph:
    passes = [
        SimplifySliceOpPass,
        BindOptionalArgsPass,
    ]
