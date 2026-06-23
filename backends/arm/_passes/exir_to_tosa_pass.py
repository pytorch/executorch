# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.arm.tosa.dialect  # noqa: F401
from executorch.backends.arm._passes.aten_to_tosa_activation_functions import (
    rewrite_clamp,
    rewrite_erf,
    rewrite_sigmoid,
    rewrite_tanh,
)
from executorch.backends.transforms.aten_to_dialect_pass import AtenToDialectPass
from executorch.exir.dialects._ops import ops as exir_ops


class ExirToTosaPass(AtenToDialectPass):
    """Rewrite simple EXIR ops to equivalent backend TOSA dialect ops.

    Rewrite functions are grouped by op category and registered with the shared
    ATen-to-dialect pass infrastructure.

    """


_ACTIVATION_FUNCTION_REWRITES = {
    exir_ops.edge.aten.clamp.default: rewrite_clamp,
    exir_ops.edge.aten.erf.default: rewrite_erf,
    exir_ops.edge.aten.sigmoid.default: rewrite_sigmoid,
    exir_ops.edge.aten.tanh.default: rewrite_tanh,
}

_DIRECT_REWRITE_CATEGORIES = {
    "activation_functions": _ACTIVATION_FUNCTION_REWRITES,
}

# Register each category's ATen targets with the function that builds the
# corresponding TOSA dialect node spec.
for _rewrite_category in _DIRECT_REWRITE_CATEGORIES.values():
    for _edge_target, _rewrite_fn in _rewrite_category.items():
        ExirToTosaPass.register_dialect_substitution(_edge_target)(_rewrite_fn)
