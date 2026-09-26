# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.transforms.aten_to_dialect_pass import (
    AtenToDialectPass,
    DialectNodeSpec,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Node


def rewrite_comparison_operator(
    node: Node, pass_: AtenToDialectPass
) -> DialectNodeSpec | None:
    match node.target:
        case exir_ops.edge.aten.eq.Tensor:
            target = exir_ops.backend.tosa.EQUAL.default
        case exir_ops.edge.aten.ge.Tensor:
            target = exir_ops.backend.tosa.GREATER_EQUAL.default
        case exir_ops.edge.aten.gt.Tensor:
            target = exir_ops.backend.tosa.GREATER.default
        case _:
            return None

    return DialectNodeSpec(target, node.args, dict(node.kwargs))
