# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import cast, Literal, Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmOpTargetedPass
from executorch.backends.arm._passes.decompose_div_pass import DecomposeDivPass
from executorch.backends.arm.tosa.specification import get_context_spec
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

edge_div_mode_ops = (exir_ops.edge.aten.div.Tensor_mode,)
aten_div_mode_ops = (torch.ops.aten.div.Tensor_mode,)
RoundingMode = Literal["trunc", "floor"]

edge_unary = {
    "div": exir_ops.edge.aten.div.Tensor,
    "floor": exir_ops.edge.aten.floor.default,
    "ceil": exir_ops.edge.aten.ceil.default,
    "eq": exir_ops.edge.aten.eq.Tensor,
    "full": exir_ops.edge.aten.full.default,
    "gt": exir_ops.edge.aten.gt.Tensor,
    "logical_and": exir_ops.edge.aten.logical_and.default,
    "logical_not": exir_ops.edge.aten.logical_not.default,
    "logical_xor": exir_ops.edge.aten.logical_xor.default,
    "intdiv": exir_ops.backend.tosa.INTDIV.default,
    "mul": exir_ops.edge.aten.mul.Tensor,
    "sub": exir_ops.edge.aten.sub.Tensor,
    "to": exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
    "where": exir_ops.edge.aten.where.self,
}

aten_unary = {
    "div": torch.ops.aten.div.Tensor,
    "floor": torch.ops.aten.floor.default,
    "ceil": torch.ops.aten.ceil.default,
    "eq": torch.ops.aten.eq.Tensor,
    "full": torch.ops.aten.full.default,
    "gt": torch.ops.aten.gt.Tensor,
    "logical_and": torch.ops.aten.logical_and.default,
    "logical_not": torch.ops.aten.logical_not.default,
    "logical_xor": torch.ops.aten.logical_xor.default,
    "mul": torch.ops.aten.mul.Tensor,
    "sub": torch.ops.aten.sub.Tensor,
    "to": torch.ops.aten.to.dtype,
    "where": torch.ops.aten.where.self,
}


def _get_opset(op):
    if op in edge_div_mode_ops:
        return edge_unary
    if op in aten_div_mode_ops:
        return aten_unary
    raise RuntimeError(f"div.Tensor_mode not supported for op {op}")


class DecomposeDivTensorModePass(ArmOpTargetedPass):
    """Rewrites aten.div.Tensor_mode into supported arithmetic ops.

    Floating-point flow:
        rounding_mode=None -> div(a, b)
        rounding_mode="floor" -> floor(div(a, b))
        rounding_mode="trunc" -> where(
            div(a, b) < 0,
            ceil(div(a, b)),
            floor(div(a, b)),
        )

    Integer flow:
        During transform-for-annotation, keep div.Tensor_mode intact, don't quantize it.
        During backend lowering, rewrite the div to a TOSA INTDIV (corresponding to trunc rounding_mode)
        + correcting factor for floor mode.

    """

    _passes_required_after: Set[Type[ExportPass]] = {DecomposeDivPass}
    target_ops = edge_div_mode_ops + aten_div_mode_ops
    check_allowed_to_transform = True

    def _is_integer_tensor(self, arg) -> bool:
        data = getattr(arg, "data", None)
        if data is not None:
            return arg.data.dtype in {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }
        return isinstance(arg, int)

    def _cast(self, opset, arg, dtype: torch.dtype, meta):
        if isinstance(arg, int):
            if dtype.is_floating_point:
                return float(arg)
            else:
                return arg
        if isinstance(arg, float):
            if dtype.is_floating_point:
                return arg
            else:
                return int(arg)
        data = getattr(arg, "data", None)
        if data is not None and data.dtype == dtype:
            return arg
        return super().call_operator(
            opset["to"],
            (arg,),
            {"dtype": dtype},
            meta,
            updated=True,
        )

    def _full(self, opset, value, dtype: torch.dtype, meta):
        return super().call_operator(
            opset["full"],
            args=((1,) * len(meta["val"].size()), value),
            kwargs={"dtype": dtype, "device": meta["val"].device},
            meta=meta,
            updated=True,
        )

    def _correct_intdiv_floor(
        self, opset, numerator, denominator, trunced_quotient, meta
    ):
        """Apply a correcting factor for converting the truncated division to
        floored division.

        Done by subtracting one from the result when, elementwise,
            - The remainder is nonzero (otherwise the division is even and the rounding trivial)
            - The numerator and denominator have different signs (causing a negative quotient)
        The sign of the quotient can't be checked directly, there are cases when it is 0 and still needs correction.

        """
        # Condition 1: non-zero remainder
        product = super().call_operator(
            opset["mul"], (trunced_quotient, denominator), {}, meta, updated=True
        )
        remainder = super().call_operator(
            opset["sub"], (numerator, product), {}, meta, updated=True
        )
        zero = self._full(opset, 0, torch.int32, meta)
        remainder_is_zero = super().call_operator(
            opset["eq"], (remainder, zero), {}, meta, updated=True
        )
        remainder_is_nonzero = super().call_operator(
            opset["logical_not"], (remainder_is_zero,), {}, meta, updated=True
        )
        # Condition 2: un-rounded quotient is negative
        a_is_negative = super().call_operator(
            opset["gt"], (zero, numerator), {}, meta, updated=True
        )
        b_is_negative = super().call_operator(
            opset["gt"], (zero, denominator), {}, meta, updated=True
        )
        signs_differ = super().call_operator(
            opset["logical_xor"],
            (a_is_negative, b_is_negative),
            {},
            meta,
            updated=True,
        )
        # Use conditions to correct quotient.
        needs_correction = super().call_operator(
            opset["logical_and"],
            (remainder_is_nonzero, signs_differ),
            {},
            meta,
            updated=True,
        )
        # (TOSA spec enforces that int(bool_var) == 1 ? bool_var : 0)
        correction = self._cast(opset, needs_correction, torch.int32, meta)
        return super().call_operator(
            opset["sub"], (trunced_quotient, correction), {}, meta, updated=True
        )

    def _call_integer_div(self, opset, a, b, rounding_mode: RoundingMode, meta):
        """Cast inputs to int32, do TOSA INTDIV, and apply correcting factor for
        floor rounding mode.
        """

        a_int32 = self._cast(opset, a, torch.int32, meta)
        b_int32 = self._cast(opset, b, torch.int32, meta)
        intdiv = super().call_operator(
            opset["intdiv"],
            (a_int32, b_int32),
            {},
            meta,
            updated=True,
        )
        if rounding_mode == "floor":
            intdiv = self._correct_intdiv_floor(opset, a_int32, b_int32, intdiv, meta)

        output_dtype = meta["val"].dtype
        return self._cast(opset, intdiv, output_dtype, meta)

    def _call_fp_div(self, opset, a, b, rounding_mode: RoundingMode | None, meta):
        q = super().call_operator(opset["div"], (a, b), {}, meta, updated=True)

        match rounding_mode:
            case None:
                return q
            case "floor":
                return super().call_operator(
                    opset["floor"], (q,), {}, meta, updated=True
                )
            case "trunc":
                zero = self._full(opset, 0.0, torch.float32, meta)
                is_neg = super().call_operator(
                    opset["gt"], (zero, q), {}, meta, updated=True
                )
                ceilq = super().call_operator(
                    opset["ceil"], (q,), {}, meta, updated=True
                )
                floorq = super().call_operator(
                    opset["floor"], (q,), {}, meta, updated=True
                )
                return super().call_operator(
                    opset["where"], (is_neg, ceilq, floorq), {}, meta, updated=True
                )

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.target_ops or not self.allowed_to_transform(meta):
            return super().call_operator(op, args, kwargs, meta)

        opset = _get_opset(op)

        a, b = args[0], args[1]
        a_is_int = self._is_integer_tensor(a)
        b_is_int = self._is_integer_tensor(b)
        rounding_mode = kwargs.get("rounding_mode", None)
        if rounding_mode is None and len(args) > 2:
            rounding_mode = args[2]
        if rounding_mode not in ("floor", "trunc", None):
            raise RuntimeError(
                "Integer div.Tensor_mode requires rounding_mode floor, trunc, or None."
                f"got {rounding_mode!r}"
            )
        rounding_mode = cast(RoundingMode | None, rounding_mode)

        int_operation = rounding_mode is not None and a_is_int and b_is_int
        sufficient_int_support = (
            rounding_mode == "trunc" or get_context_spec().support_integer()
        )
        sufficient_int_support &= not get_context_spec().is_U55_subset

        if int_operation and sufficient_int_support:
            """Integer operation and necessary int ops supported -> pure integer
            path.
            """
            if self.is_tfa_pass:
                # No quantization neccessary, so don't do anything in TFA.
                return super().call_operator(op, args, kwargs, meta)
            return self._call_integer_div(opset, a, b, rounding_mode, meta)
        else:
            """Otherwise floating point operation -> do fp path.

            Cast to and from fp if neccessary.

            """
            if a_is_int:
                a = self._cast(opset, a, torch.float32, meta)
            if b_is_int:
                b = self._cast(opset, b, torch.float32, meta)

            result = self._call_fp_div(
                opset,
                a,
                b,
                rounding_mode,
                meta,
            )

            output_dtype = meta["val"].dtype
            if output_dtype != torch.float32:
                result = self._cast(opset, result, output_dtype, meta)

            return result
