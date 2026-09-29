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


def rewrite_ternary_operator(
    node: Node, pass_: AtenToDialectPass
) -> DialectNodeSpec | None:
    match node.target:
        case exir_ops.edge.aten.where.self:
            return DialectNodeSpec(
                exir_ops.backend.tosa.SELECT.default,
                node.args,
                dict(node.kwargs),
            )
        case _:
            return None
