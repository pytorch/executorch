# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import get_node_arg
from executorch.backends.arm._passes.fuse_constant_ops_pass import ComputeConstantOpsAOT
from executorch.backends.arm._passes.insert_table_ops import InsertTableOpsPass
from executorch.backends.arm._passes.match_arg_dtype_pass import MatchArgDtypePass
from executorch.backends.arm._passes.match_arg_ranks_pass import MatchArgRanksPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

torch_gelu = (torch.ops.aten.gelu.default,)

edge_gelu = (exir_ops.edge.aten.gelu.default,)


def _get_gelu_ops(op) -> tuple:
    """
    Returns the operators needed to decompose GELU
    """

    if op in edge_gelu:
        return (
            exir_ops.edge.aten.full.default,
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.edge.aten.tanh.default,
            exir_ops.edge.aten.erf.default,
        )
    if op in torch_gelu:
        return (
            torch.ops.aten.full.default,
            torch.ops.aten.add.Tensor,
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.tanh.default,
            torch.ops.aten.erf.default,
        )
    raise RuntimeError(f"Can't get GeLU decomposition ops for op {op}")


class DecomposeGeluPass(ArmPass):
    """
    This pass decomposes the GELU operator into primitive ops.
    Aiming to adhere closely to the reference implementations built into
    ExecuTorch. Including using the same pre-calculated constants.

    This operator has two formulae depending on the value of the
    approximate argument. Examples below include the added full
    operators necessary for the initialization for constants used in
    each respective formula.

    aten.gelu(x, approximate="none") becomes:
        %FULL_0_5 = full()
        %FULL_1 = full()
        %FULL_SQRT1_2 = full()
        %op1 = mul(x, %FULL_SQRT1_2)
        %op2 = erf(%op1)
        %op3 = add(%op2, %FULL_1)
        %op4 = mul(%op3, %FULL_0_5)
        %op5 = mul(%x, %op4)

    aten.gelu(x, approximate="tanh") becomes:
        %FULL_0_5 = full()
        %FULL_1 = full()
        %FULL_SQRT2 = full()
        %FULL_2_SQRTPI = full()
        %FULL_CUBE_COEFF = full()
        %SQRT_MUL = mul(%FULL_SQRT2, %FULL_2_SQRTPI)
        %SQRT_2_PI = mul(%SQRT_MUL, %FULL_0_5)
        %sqr_x = mul(x, x)
        %cube_x = mul(sqr_x, x)
        %op1 = mul(%cube_x, %FULL_CUBE_COEFF)
        %op2 = add(%x, %op1)
        %op3 = mul(%op2, %SQRT_2_PI)
        %op4 = tanh(%op3)
        %op5 = add(%op4, %FULL_1)
        %op6 = mul(%x, %op5)
        %op7 = mul(%op6, %FULL_0_5)
    """

    _passes_required_after: Set[Type[ExportPass]] = {
        ComputeConstantOpsAOT,
        InsertTableOpsPass,
        MatchArgDtypePass,
        MatchArgRanksPass,
    }

    def call_operator(self, op, args, kwargs, meta):
        if op not in torch_gelu + edge_gelu:
            return super().call_operator(op, args, kwargs, meta)

        full_op, add_op, mul_op, tanh_op, erf_op = _get_gelu_ops(op)

        input = get_node_arg(args, 0)
        # If approximate is default (none) it does not appear in kwargs
        approximate = get_node_arg(kwargs, "approximate", "none")

        shape = meta["val"].size()
        dtype = meta["val"].dtype

        FULL_0_5 = super().call_operator(
            full_op, ([1] * len(shape), 0.5), {"dtype": dtype}, meta
        )
        FULL_1 = super().call_operator(
            full_op, ([1] * len(shape), 1), {"dtype": dtype}, meta
        )

        if approximate == "none":
            # Constant mirrors ExecuTorch implementation for parity.
            FULL_SQRT1_2 = super().call_operator(
                full_op, ([1] * len(shape), 0.70710678118654752440), {}, meta
            )

            op1 = super().call_operator(mul_op, (input, FULL_SQRT1_2), {}, meta)
            op2 = super().call_operator(erf_op, (op1,), {}, meta)
            op3 = super().call_operator(add_op, (op2, FULL_1), {}, meta)
            op4 = super().call_operator(mul_op, (op3, FULL_0_5), {}, meta)
            return super().call_operator(mul_op, (input, op4), {}, meta)

        elif approximate == "tanh":
            # Constants mirror ExecuTorch implementation for parity.
            FULL_SQRT2 = super().call_operator(
                full_op,
                ([1] * len(shape), 1.41421356237309504880),
                {"dtype": dtype},
                meta,
            )
            FULL_2_SQRTPI = super().call_operator(
                full_op,
                ([1] * len(shape), 1.12837916709551257390),
                {"dtype": dtype},
                meta,
            )
            FULL_CUBE_COEFF = super().call_operator(
                full_op, ([1] * len(shape), 0.044715), {"dtype": dtype}, meta
            )

            # Mirrors ExecuTorch implementations for calculating this value
            SQRT_MUL = super().call_operator(
                mul_op, (FULL_SQRT2, FULL_2_SQRTPI), {}, meta
            )
            SQRT_2_PI = super().call_operator(mul_op, (SQRT_MUL, FULL_0_5), {}, meta)

            # Avoiding using POW in order to reduce pass order reliance.
            sqr_x = super().call_operator(mul_op, (input, input), {}, meta)
            cube_x = super().call_operator(mul_op, (sqr_x, input), {}, meta)
            op1 = super().call_operator(mul_op, (cube_x, FULL_CUBE_COEFF), {}, meta)
            op2 = super().call_operator(add_op, (input, op1), {}, meta)
            op3 = super().call_operator(mul_op, (op2, SQRT_2_PI), {}, meta)
            op4 = super().call_operator(tanh_op, (op3,), {}, meta)
            op5 = super().call_operator(add_op, (op4, FULL_1), {}, meta)
            op6 = super().call_operator(mul_op, (input, op5), {}, meta)
            return super().call_operator(mul_op, (op6, FULL_0_5), {}, meta)
        else:
            raise RuntimeError(
                f"approximate argument expected 'none' or 'tanh' but got {approximate}"
            )
