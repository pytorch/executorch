# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops


# For FP case
edge_logit = exir_ops.edge.aten.logit.default
# For INT case
aten_logit = torch.ops.aten.logit.default


def get_ops(op):
    """Returns the appropriate operator functions based on the input operator."""
    if op == edge_logit:
        return (
            exir_ops.edge.aten.log.default,
            exir_ops.edge.aten.add.Scalar,
            exir_ops.edge.aten.reciprocal.default,
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.edge.aten.mul.Scalar,
            exir_ops.edge.aten.clamp.default,
        )
    elif op == aten_logit:
        return (
            torch.ops.aten.log.default,
            torch.ops.aten.add.Scalar,
            torch.ops.aten.reciprocal.default,
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.mul.Scalar,
            torch.ops.aten.clamp.default,
        )
    else:
        raise ValueError(f"Unsupported operator: {op}")


class DecomposeLogitPass(ArmPass):
    """
    Decomposes the `logit` operator into a sequence of primitive operations.

    If `eps` is provided, the input tensor `x` is first clamped to the range
    [eps, 1 - eps].

    The decomposition follows the identity:

        logit(x) = log(x / (1 - x))

    Examples:

        logit(x) becomes:
            log(x * reciprocal((-1) * x + 1))

        logit(x, eps) becomes:
            y = clamp(x, eps, 1 - eps)
            log(y * reciprocal((-1) * y + 1))
    """

    def call_operator(self, op, args, kwargs, meta):
        if op not in [edge_logit, aten_logit]:
            return super().call_operator(op, args, kwargs, meta)

        X = args[0]
        eps = args[1] if len(args) > 1 else kwargs.get("eps", None)

        (
            log_op,
            add_scalar_op,
            recip_op,
            mul_tensor_op,
            mul_scalar_op,
            clamp_op,
        ) = get_ops(op)

        if eps is not None:
            X = super().call_operator(
                clamp_op, (X, eps, 1.0 - eps), {}, meta, updated=True
            )

        neg_X = super().call_operator(mul_scalar_op, (X, -1.0), {}, meta, updated=True)

        denom = super().call_operator(
            add_scalar_op, (neg_X, 1.0), {}, meta, updated=True
        )

        frac = super().call_operator(recip_op, (denom,), {}, meta, updated=True)

        log_input = super().call_operator(
            mul_tensor_op, (X, frac), {}, meta, updated=True
        )

        return super().call_operator(log_op, (log_input,), {}, meta, updated=True)
