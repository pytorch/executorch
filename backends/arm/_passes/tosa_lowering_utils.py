# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Build explicit TOSA operands during lowering."""

import torch
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Graph, Node


def create_const(graph: Graph, values, dtype: torch.dtype, from_node: Node) -> Node:
    if isinstance(values, torch.Tensor):
        values = values.flatten().tolist()
    elif not isinstance(values, (tuple, list)):
        values = [values]
    values = list(values)
    node = create_node(
        graph,
        exir_ops.backend.tosa.CONST.default,
        (values, [len(values)], dtype),
        from_node=from_node,
    )
    node.meta.pop(TosaSpecialDtype.meta_key(), None)
    fake = from_node.meta["val"]
    node.meta["val"] = fake.new_empty((len(values),), dtype=dtype)
    return node
