# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

from executorch.backends.transforms.aten_to_dialect_pass import (
    AtenToDialectPass,
    DialectNodeSpec,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Node


def rewrite_argmax(node: Node, pass_: AtenToDialectPass) -> DialectNodeSpec:
    input_node = cast(Node, node.args[0])
    dim = cast(int, node.kwargs["dim"] if "dim" in node.kwargs else node.args[1])
    if dim < 0:
        dim += len(input_node.meta["val"].shape)

    return DialectNodeSpec(
        exir_ops.backend.tosa.ARGMAX.default,
        (input_node, dim),
        {},
    )
