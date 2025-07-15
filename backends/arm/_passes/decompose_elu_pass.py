# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops

edge_elu_ops = (exir_ops.edge.aten.elu.default,)
aten_elu_ops = (torch.ops.aten.elu.default, torch.ops.aten.elu_.default)


def get_elu_decomposition(op) -> tuple:
    """
    Returns the decomposition of the given aten.elu operation into
    its equivalent TOSA-supported operations

    This handles both edge dialect ops and core PyTorch ops. The decomposition strategy
    is:
        elu(x, y) → where(greater_or_eq(x, 0), (exp(x)-1), x)

    Returns:
        A tuple (exp_op, sub_op, ge_op, where_op) corresponding to the appropriate operator
        overloads for the input op.

    Raises:
        RuntimeError: If the provided operator is not a supported elu variant.
    """

    if op in edge_elu_ops:
        return (
            exir_ops.edge.aten.add.Scalar,
            exir_ops.edge.aten.exp.default,
            exir_ops.edge.aten.ge.Scalar,
            exir_ops.edge.aten.where.self,
            exir_ops.edge.aten.mul.Scalar,
        )

    if op in aten_elu_ops:
        return (
            torch.ops.aten.add.Scalar,
            torch.ops.aten.exp.default,
            torch.ops.aten.ge.Scalar,
            torch.ops.aten.where.self,
            torch.ops.aten.mul.Scalar,
        )

    raise RuntimeError(f"Can't get elu decomposition for op {op}")


class DecomposeEluPass(ArmPass):
    """
    A transformation pass that decomposes unsupported 'aten.elu' operations
    into a combination of supported TOSA-equivalent operations.

    Since TOSA does not provide a native ELU operator, this pass rewrites:
        elu(x) → where(greater_or_eq(x, 0), (alpha*(exp(x)-1)), x)

    Supported input ops:
        - aten.elu(x)
        - aten.elu_(x)
        - exir_ops.edge.aten.elu.Tensor(x)

    These are replaced with:
        - aten.exp or exir_ops.edge.aten.exp
        - aten.sub.Scalar or exir_ops.edge.aten.sub.Scalar
        - aten.ge.Scalar or exir_ops.edge.aten.ge.Scalar
        - aten.where.self or exir_ops.edge.aten.where.self
        - aten.mul.Scalar or exir_ops.edge.aten.mul.Scalar
    """

    def call_operator(self, op, args, kwargs, meta):
        if op not in (edge_elu_ops + aten_elu_ops):
            return super().call_operator(op, args, kwargs, meta, updated=False)

        (
            add_op,
            exp_op,
            ge_op,
            where_op,
            mul_op,
        ) = get_elu_decomposition(op)

        input = args[0]
        alpha = int(args[1]) if len(args) > 1 else 1

        exp_node = super().call_operator(exp_op, (input,), {}, meta, updated=True)
        sub_node = super().call_operator(
            add_op, (exp_node, -1.0), {}, meta, updated=True
        )
        mul_node = super().call_operator(
            mul_op, (sub_node, alpha), {}, meta, updated=True
        )
        ge_node = super().call_operator(ge_op, (input, 0.0), {}, meta, updated=True)
        where_node = super().call_operator(
            where_op, (ge_node, input, mul_node), {}, meta, updated=True
        )

        return where_node
