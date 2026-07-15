# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable

import executorch.backends.arm.tosa.dialect  # noqa: F401
from executorch.backends.arm._passes.aten_to_tosa_activation_functions import (
    get_activation_replacement,
)
from executorch.backends.arm._passes.aten_to_tosa_tensor_operators import (
    rewrite_argmax,
    rewrite_binary_operator,
    rewrite_rfft2,
    rewrite_unary_operator,
)
from executorch.backends.transforms.aten_to_dialect_pass import (
    AtenToDialectPass,
    DialectNodeSpec,
    SubstitutionFn,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Node
from torch.fx.node import Target


class ExirToTosaPass(AtenToDialectPass):
    """Rewrite simple EXIR ops to equivalent backend TOSA dialect ops.

    Rewrite functions are registered with the shared ATen-to-dialect pass
    infrastructure.

    """


def register_dialect_substitutions(
    *targets: Target,
) -> Callable[[SubstitutionFn], SubstitutionFn]:
    def decorator(func: SubstitutionFn) -> SubstitutionFn:
        for target in targets:
            ExirToTosaPass.register_dialect_substitution(target)(func)
        return func

    return decorator


@register_dialect_substitutions(
    exir_ops.edge.aten.argmax.default,
)
def _get_tensor_operators_replacement(
    node: Node, pass_: AtenToDialectPass
) -> DialectNodeSpec | None:
    return rewrite_argmax(node, pass_)


@register_dialect_substitutions(
    exir_ops.edge.aten.fft_rfft2.default,
)
def _get_fft_replacement(
    node: Node, pass_: AtenToDialectPass
) -> DialectNodeSpec | None:
    return rewrite_rfft2(node, pass_)


@register_dialect_substitutions(
    exir_ops.edge.aten.add.Tensor,
    exir_ops.edge.aten.bitwise_and.Tensor,
    exir_ops.edge.aten.bitwise_left_shift.Tensor,
    exir_ops.edge.aten.bitwise_or.Tensor,
    exir_ops.edge.aten.bitwise_right_shift.Tensor,
    exir_ops.edge.aten.bitwise_xor.Tensor,
    exir_ops.edge.aten.eq.Tensor,
    exir_ops.edge.aten.ge.Tensor,
    exir_ops.edge.aten.gt.Tensor,
    exir_ops.edge.aten.logical_and.default,
    exir_ops.edge.aten.logical_or.default,
    exir_ops.edge.aten.logical_xor.default,
    exir_ops.edge.aten.maximum.default,
    exir_ops.edge.aten.minimum.default,
    exir_ops.edge.aten.mul.Tensor,
    exir_ops.edge.aten.pow.Tensor_Tensor,
    exir_ops.edge.aten.sub.Tensor,
)
def _get_binary_operator_replacement(
    node: Node, pass_: AtenToDialectPass
) -> DialectNodeSpec | None:
    return rewrite_binary_operator(node, pass_)


@register_dialect_substitutions(
    exir_ops.edge.aten.abs.default,
    exir_ops.edge.aten.bitwise_not.default,
    exir_ops.edge.aten.ceil.default,
    exir_ops.edge.aten.cos.default,
    exir_ops.edge.aten.exp.default,
    exir_ops.edge.aten.floor.default,
    exir_ops.edge.aten.log.default,
    exir_ops.edge.aten.logical_not.default,
    exir_ops.edge.aten.neg.default,
    exir_ops.edge.aten.reciprocal.default,
    exir_ops.edge.aten.rsqrt.default,
    exir_ops.edge.aten.sin.default,
)
def _get_unary_operator_replacement(
    node: Node, pass_: AtenToDialectPass
) -> DialectNodeSpec | None:
    return rewrite_unary_operator(node, pass_)


@register_dialect_substitutions(
    exir_ops.edge.aten.clamp.default,
    exir_ops.edge.aten.erf.default,
    exir_ops.edge.aten.sigmoid.default,
    exir_ops.edge.aten.tanh.default,
)
def _get_activation_replacement(
    node: Node, pass_: AtenToDialectPass
) -> DialectNodeSpec | None:
    return get_activation_replacement(node, pass_)
