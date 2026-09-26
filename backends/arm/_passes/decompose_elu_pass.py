# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

edge_elu_ops = (exir_ops.edge.aten.elu.default,)
edge_selu_ops = (exir_ops.edge.aten.selu.default,)
edge_celu_ops = (exir_ops.edge.aten.celu.default,)
edge_elu_family_ops = edge_elu_ops + edge_selu_ops + edge_celu_ops
torch_selu_ops = (torch.ops.aten.selu.default,)
torch_celu_ops = (torch.ops.aten.celu.default,)
selu_ops = edge_selu_ops + torch_selu_ops
celu_ops = edge_celu_ops + torch_celu_ops

SELU_ALPHA = 1.6732632423543772
SELU_SCALE = 1.0507009873554805


def get_elu_decomposition(op) -> tuple:
    """Returns the decomposition of the given aten.elu operation into its
    equivalent TOSA-supported operations.

    This handles both edge dialect ops and core PyTorch ops. The decomposition strategy
    is:
        elu(x, y) → where(greater_or_eq(x, 0), (exp(x)-1), x)

    Returns:
        A tuple (expm1_op, ge_op, where_op, mul_op) corresponding to the appropriate operator
        overloads for the input op.

    Raises:
        RuntimeError: If the provided operator is not a supported elu variant.

    """

    if op in edge_elu_family_ops:
        return (
            exir_ops.edge.aten.expm1.default,
            exir_ops.edge.aten.ge.Scalar,
            exir_ops.edge.aten.where.self,
            exir_ops.edge.aten.mul.Scalar,
        )

    raise RuntimeError(f"Can't get elu decomposition for op {op}")


def _get_elu_parameter(args, kwargs, index, name):
    if len(args) > index:
        return args[index]

    return kwargs.get(name, 1.0)


def _get_elu_parameters(op, args, kwargs):
    if op in selu_ops:
        return SELU_ALPHA, SELU_SCALE, 1.0
    if op in celu_ops:
        alpha = _get_elu_parameter(args, kwargs, 1, "alpha")
        return alpha, 1.0, 1.0 / alpha

    alpha = _get_elu_parameter(args, kwargs, 1, "alpha")
    scale = _get_elu_parameter(args, kwargs, 2, "scale")
    input_scale = _get_elu_parameter(args, kwargs, 3, "input_scale")
    return alpha, scale, input_scale


class ConvertEluFamilyToEluPass(ArmOpTargetedPass):
    """Convert SELU/CELU ops to equivalent parameterized ELU ops."""

    _passes_required_after: Set[Type[ExportPass]] = set()
    target_ops = selu_ops + celu_ops
    check_allowed_to_transform = True

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.target_ops or not self.allowed_to_transform(meta):
            return super().call_operator(op, args, kwargs, meta, updated=False)

        input_ = args[0]
        alpha, scale, input_scale = _get_elu_parameters(op, args, kwargs)
        elu_op = (
            torch.ops.aten.elu.default
            if op in torch_selu_ops + torch_celu_ops
            else exir_ops.edge.aten.elu.default
        )
        return super().call_operator(
            elu_op,
            (input_, alpha, scale, input_scale),
            {},
            meta,
            updated=True,
        )


class DecomposeEluPass(ArmOpTargetedPass):
    """A transformation pass that decomposes unsupported 'aten.elu' operations
    into a combination of supported TOSA-equivalent operations.

    Since TOSA does not provide a native ELU operator, this pass rewrites:
        elu(x) → scale * where(
            greater_or_eq(x, 0), x, alpha * expm1(input_scale * x)
        )

    Supported input ops:
        - exir_ops.edge.aten.elu.default
        - exir_ops.edge.aten.selu.default
        - exir_ops.edge.aten.celu.default

    These are replaced with:
        - exir_ops.edge.aten.expm1.default
        - exir_ops.edge.aten.ge.Scalar
        - exir_ops.edge.aten.where.self
        - exir_ops.edge.aten.mul.Scalar

    """

    _passes_required_after: Set[Type[ExportPass]] = set()
    target_ops = edge_elu_family_ops

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.target_ops:
            return super().call_operator(op, args, kwargs, meta, updated=False)

        if self._is_quantized_meta(meta):
            # If quantized, node should be replace by table op
            return super().call_operator(op, args, kwargs, meta)

        (
            expm1_op,
            ge_op,
            where_op,
            mul_op,
        ) = get_elu_decomposition(op)

        input = args[0]
        alpha, scale, input_scale = _get_elu_parameters(op, args, kwargs)

        if alpha == 0:
            relu_op = exir_ops.edge.aten.clamp.default
            relu_node = super().call_operator(
                relu_op,
                (
                    input,
                    0,
                ),
                {},
                meta,
                updated=True,
            )
            if scale == 1:
                return relu_node

            return super().call_operator(
                mul_op, (relu_node, scale), {}, meta, updated=True
            )

        expm1_input = input
        if input_scale != 1:
            expm1_input = super().call_operator(
                mul_op, (input, input_scale), {}, meta, updated=True
            )
        expm1_node = super().call_operator(
            expm1_op, (expm1_input,), {}, meta, updated=True
        )
        mul_node = super().call_operator(
            mul_op, (expm1_node, alpha), {}, meta, updated=True
        )
        ge_node = super().call_operator(ge_op, (input, 0.0), {}, meta, updated=True)
        positive_node = input
        if scale != 1:
            positive_node = super().call_operator(
                mul_op, (input, scale), {}, meta, updated=True
            )
            mul_node = super().call_operator(
                mul_op, (mul_node, scale), {}, meta, updated=True
            )
        where_node = super().call_operator(
            where_op, (ge_node, positive_node, mul_node), {}, meta, updated=True
        )

        return where_node
