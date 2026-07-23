# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional, Set, Type

from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class DecomposeIntPowPass(ArmOpTargetedPass):
    """Replaces pow with integer exponent with a series of multiplications.

    Only handles pow.Tensor_Scalar and not pow.Tensor_Tensor. Needs to be run
    before doing scalar to tensor conversion.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()
    target_ops = (exir_ops.edge.aten.pow.Tensor_Scalar,)

    targeted_ops = {exir_ops.edge.aten.pow.Tensor_Scalar}

    @staticmethod
    def _get_decomposable_integer_exponent(exp) -> Optional[int]:
        if isinstance(exp, int):
            return exp
        # Exported models can represent positive integer-valued exponents as
        # floats, for example pow(x, 2.0). Only exact values are decomposed:
        # rounding near-integer floats would change fractional pow semantics,
        # especially for negative bases.
        if isinstance(exp, float) and exp > 0 and exp.is_integer():
            return int(exp)
        return None

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.target_ops:
            return super().call_operator(op, args, kwargs, meta)

        if self._is_quantized_meta(meta):
            # If quantized, node should be replace by table op
            return super().call_operator(op, args, kwargs, meta)

        x = args[0]
        exp = args[1]

        if exp == 0:
            zeros = super().call_operator(
                exir_ops.edge.aten.sub.Tensor, (x, x), {}, meta, True
            )
            ones = super().call_operator(
                exir_ops.edge.aten.full_like.default, (x, 1), {}, meta, True
            )
            return super().call_operator(
                exir_ops.edge.aten.add.Tensor, (zeros, ones), {}, meta, True
            )

        exp = self._get_decomposable_integer_exponent(exp)
        if exp is None:
            return super().call_operator(op, args, kwargs, meta)

        if exp == 1:
            ones = super().call_operator(
                exir_ops.edge.aten.full_like.default, (x, 1), {}, meta, True
            )
            return super().call_operator(
                exir_ops.edge.aten.mul.Tensor, (x, ones), {}, meta, True
            )

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
