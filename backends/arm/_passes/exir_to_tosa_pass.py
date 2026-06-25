# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.arm.tosa.dialect  # noqa: F401
from executorch.backends.arm._passes.aten_to_tosa_activation_functions import (
    get_activation_replacement,
)
from executorch.backends.arm._passes.aten_to_tosa_tensor_operators import rewrite_argmax
from executorch.backends.transforms.aten_to_dialect_pass import (
    AtenToDialectPass,
    DialectNodeSpec,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Node


class ExirToTosaPass(AtenToDialectPass):
    """Rewrite simple EXIR ops to equivalent backend TOSA dialect ops.

    Rewrite functions are registered with the shared ATen-to-dialect pass
    infrastructure.

    """


@ExirToTosaPass.register_dialect_substitution(exir_ops.edge.aten.argmax.default)
def _get_tensor_operators_replacement(
    node: Node, pass_: AtenToDialectPass
) -> DialectNodeSpec:
    return rewrite_argmax(node, pass_)


@ExirToTosaPass.register_dialect_substitution(exir_ops.edge.aten.clamp.default)
@ExirToTosaPass.register_dialect_substitution(exir_ops.edge.aten.erf.default)
@ExirToTosaPass.register_dialect_substitution(exir_ops.edge.aten.sigmoid.default)
@ExirToTosaPass.register_dialect_substitution(exir_ops.edge.aten.tanh.default)
def _get_activation_replacement(
    node: Node, pass_: AtenToDialectPass
) -> DialectNodeSpec | None:
    return get_activation_replacement(node, pass_)
