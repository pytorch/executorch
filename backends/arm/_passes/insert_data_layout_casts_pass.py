# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm.tosa.specification import get_context_spec
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, NodeMetadata


class InsertDataLayoutCastsPass(ArmPass):
    """Insert casts around data layout operators when their dtype is not
    supported by the active TOSA specification.

    This pass targets operators that lower to TOSA data layout operators:
    CONCAT, PAD, RESHAPE, REVERSE, SLICE, TILE, and TRANSPOSE.

    Example:
        Before pass:
            y = transpose(x)  # x.data.dtype == torch.int32
        After pass:
            xfp32 = _to_dim_order_copy(x, dtype=torch.float32)
            yfp32 = transpose(xfp32)
            y = _to_dim_order_copy(yfp32, dtype=torch.int32)

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    _cast_op = exir_ops.edge.dim_order_ops._to_dim_order_copy.default

    _concat_ops = {
        exir_ops.edge.aten.cat.default,
        exir_ops.edge.aten.concatenate.default,
    }
    _single_input_ops = {
        exir_ops.backend.tosa.TRANSPOSE.default,
        exir_ops.edge.aten.constant_pad_nd.default,
        exir_ops.edge.aten.view_copy.default,
        exir_ops.edge.aten.repeat.default,
        exir_ops.edge.aten.permute_copy.default,
        exir_ops.edge.aten.slice_copy.Tensor,
        exir_ops.edge.aten.flip.default,
    }
    targeted_ops = _concat_ops | _single_input_ops

    _fp_to_int_map = {
        torch.float16: torch.int16,
        torch.bfloat16: torch.int16,
        torch.float32: torch.int32,
    }

    _int_to_fp_map = {
        torch.int8: torch.float16,  # This doubles the size after casting, but is very unlikely to occur in practice since int8 is only ever used by LOGICAL_SHIFT and CAST/RESCALE ops in PRO-FP.
        torch.int16: torch.float16,
        torch.int32: torch.float32,
    }

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.targeted_ops:
            return super().call_operator(op, args, kwargs, meta)

        if op in self._concat_ops:
            # Cast to largest dtype
            dtypes = [arg.data.dtype for arg in args[0]]
            dtype_sizes = [dtype.itemsize for dtype in dtypes]
            dtype = dtypes[dtype_sizes.index(max(dtype_sizes))]
        else:
            dtype = args[0].data.dtype

        spec = get_context_spec()
        dtype_is_integer = not dtype.is_floating_point and dtype != torch.bool
        if dtype_is_integer and not spec.support_integer():
            supported_dtype = self._int_to_fp_map.get(dtype, None)
        elif dtype.is_floating_point and not spec.support_float():
            supported_dtype = self._fp_to_int_map.get(dtype, None)
        else:
            return super().call_operator(op, args, kwargs, meta)

        # CONCATENATE does not support int16 w/o INT16 extension like other ops
        if (
            op in self._concat_ops
            and supported_dtype == torch.int16
            and not spec.support_extension("int16")
        ):
            supported_dtype = None

        if supported_dtype is None:
            raise TypeError(
                f"Data type {dtype} of operator {op} is not supported by"
                f" {spec}, and casting is currently not supported by {self.__class__.__name__}."
            )

        if op in self._concat_ops:
            x_casted = []
            for arg in args[0]:
                x_casted.append(
                    super().call_operator(
                        self._cast_op,
                        (arg,),
                        {"dtype": supported_dtype},
                        NodeMetadata(arg.node.meta),
                        updated=True,
                    )
                )
            y_casted = super().call_operator(
                op, (x_casted, *args[1:]), kwargs, meta, updated=True
            )

        else:
            x_casted = super().call_operator(
                self._cast_op,
                (args[0],),
                {"dtype": supported_dtype},
                NodeMetadata(args[0].node.meta),
                updated=True,
            )
            y_casted = super().call_operator(
                op, (x_casted, *args[1:]), kwargs, meta, updated=True
            )

        y = super().call_operator(
            self._cast_op, (y_casted,), {"dtype": dtype}, meta, updated=True
        )
        return y
