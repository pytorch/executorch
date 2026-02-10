# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torch.fx import Node


batch_norm_target_ops = [
    torch.ops.aten.batch_norm.default,
    torch.ops.aten.native_batch_norm.default,
    torch.ops.aten._native_batch_norm_legit_no_training.default,
]


def is_op_node(node: Node, target_op) -> bool:
    if isinstance(target_op, list):
        target_ops = target_op
    else:
        target_ops = [target_op]

    return (
        node is not None
        and hasattr(node, "op")
        and node.op == "call_function"
        and hasattr(node, "target")
        and node.target in target_ops
    )


def is_batch_norm(node: Node) -> bool:
    return is_op_node(node, batch_norm_target_ops)
