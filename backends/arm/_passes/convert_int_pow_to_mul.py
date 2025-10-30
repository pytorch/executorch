# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Set, Type

from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class ConvertIntPowToMuls(ArmPass):
    """
    Replaces pow with integer exponent with a series of multiplications.
    Only handles pow.Tensor_Scalar and not pow.Tensor_Tensor.
    Needs to be run before doing scalar to tensor conversion.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call_operator(self, op, args, kwargs, meta):
        if op != exir_ops.edge.aten.pow.Tensor_Scalar:
            return super().call_operator(op, args, kwargs, meta)

        x = args[0]
        exp = args[1]

        # Handle zero first and return early
        if exp == 0:
            # return a tensor of ones with the same shape as x
            return super().call_operator(
                exir_ops.edge.aten.full_like.default, (x, 1), {}, meta, True
            )

        if not isinstance(exp, int):
            return super().call_operator(op, args, kwargs, meta)

        # Handle negative exponent
        if exp < 0:
            x = super().call_operator(
                exir_ops.edge.aten.reciprocal.default, (x,), {}, meta, True
            )
            exp = -exp

        res = x

        # Consider exponentiation by squaring, if exp turns out to be large.
        # Now we just roll out the multiplications.
        for _ in range(exp - 1):
            res = super().call_operator(
                exir_ops.edge.aten.mul.Tensor, (res, x), {}, meta, True
            )

        return res
