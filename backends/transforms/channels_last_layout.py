# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.transforms.channels_last_ops  # noqa: F401

import torch

from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.node import Target

ATEN_PERMUTE_COPY = exir_ops.edge.aten.permute_copy.default
LAYOUT_PERMUTE_COPY = exir_ops.edge.channels_last.permute_copy.default
PERMUTE_COPY_TARGETS: frozenset[Target] = frozenset(
    (ATEN_PERMUTE_COPY, LAYOUT_PERMUTE_COPY)
)


def is_permute_copy(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target in PERMUTE_COPY_TARGETS


def is_layout_copy(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target == LAYOUT_PERMUTE_COPY


def is_channels_last_input_normalization_pair(
    first: torch.fx.Node, second: torch.fx.Node
) -> bool:
    if first.target != ATEN_PERMUTE_COPY or second.target != LAYOUT_PERMUTE_COPY:
        return False
    input_node = first.args[0] if first.args else None
    val = input_node.meta.get("val") if isinstance(input_node, torch.fx.Node) else None
    return (
        input_node is not None
        and input_node.op == "placeholder"
        and isinstance(val, torch.Tensor)
        and val.dim() == 4
        and tuple(val.dim_order()) == (0, 2, 3, 1)
        and list(first.args[1]) == [0, 2, 3, 1]
        and list(second.args[1]) == [0, 3, 1, 2]
    )


def composed_permute_target(first: torch.fx.Node, second: torch.fx.Node) -> Target:
    if is_layout_copy(first) and is_layout_copy(second):
        return LAYOUT_PERMUTE_COPY
    return ATEN_PERMUTE_COPY
