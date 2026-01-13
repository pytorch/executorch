# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# The TOSA BITWISE_AND, BITWISE_OR, and BITWISE_XOR don't handle bool inputs.
# When a targeted op receives boolean tensors, we promote them to an integer type before
# invocation and cast the result back to the expected dtype afterwards.

from typing import Set, Type

import torch

from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class PromoteBoolOperandsPass(ArmPass):
    """Promote boolean operands to the appropriate integer dtype for unsupported ops."""

    _passes_required_after: Set[Type[ExportPass]] = set()

    targeted_ops = {
        exir_ops.edge.aten.bitwise_and.Tensor,
        exir_ops.edge.aten.bitwise_or.Tensor,
        exir_ops.edge.aten.bitwise_xor.Tensor,
        exir_ops.edge.aten.mul.Tensor,
    }

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.targeted_ops:
            return super().call_operator(op, args, kwargs, meta)

        original_dtypes = [arg.data.dtype for arg in args]
        if torch.bool not in original_dtypes:
            return super().call_operator(op, args, kwargs, meta)

        # select the first non-bool dtype, or None if all bool
        promoted_dtype = next((dt for dt in original_dtypes if dt != torch.bool), None)

        # if we don't have a dtype specified by the op, promote to default choice for the op
        if promoted_dtype is None:
            if op == exir_ops.edge.aten.mul.Tensor:
                # mul as int32
                promoted_dtype = torch.int32
            else:
                # bitwise ops can be int8
                promoted_dtype = torch.int8

        target_dtypes = []
        for dt in original_dtypes:
            if dt == torch.bool:
                target_dtypes.append(promoted_dtype)
            else:
                target_dtypes.append(dt)

        new_args = []
        for arg, original_dtype, target_dtype in zip(
            args, original_dtypes, target_dtypes
        ):
            if original_dtype == target_dtype:
                new_args.append(arg)
            else:
                new_args.append(
                    super().call_operator(
                        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                        (arg,),
                        {"dtype": target_dtype},
                        meta,
                    )
                )

        output = super().call_operator(
            op,
            tuple(new_args),
            kwargs,
            meta,
        )

        if all(dtype == torch.bool for dtype in original_dtypes):
            output = super().call_operator(
                exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                (output,),
                {"dtype": torch.bool},
                meta,
            )
        return output
