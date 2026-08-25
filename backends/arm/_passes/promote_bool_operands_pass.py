# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Some TOSA ops don't handle bool inputs. When a targeted op receives boolean
# tensors, we promote them to an integer type before invocation and cast the
# result back to the expected dtype afterwards.

from typing import Set, Type

import torch

from executorch.backends.arm._passes.arm_pass import ArmOpTargetedPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class PromoteBoolOperandsPass(ArmOpTargetedPass):
    """Promote boolean operands to the appropriate integer dtype for unsupported
    ops.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    # Bool bitwise ops are handled by RewriteBoolBitwiseToLogicalPass. Promoting
    # them here would hide the bool dtype and prevent that rewrite.
    target_ops = {
        exir_ops.edge.aten.mul.Tensor,
    }

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.target_ops:
            return super().call_operator(op, args, kwargs, meta)

        original_dtypes = [arg.data.dtype for arg in args]
        if torch.bool not in original_dtypes:
            return super().call_operator(op, args, kwargs, meta)

        # select the first non-bool dtype, or None if all bool
        promoted_dtype = next((dt for dt in original_dtypes if dt != torch.bool), None)

        # If all operands are bool, promote mul to int32.
        if promoted_dtype is None:
            promoted_dtype = torch.int32

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
