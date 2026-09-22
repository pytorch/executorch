# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers for native backend graph passes."""

import torch
from torch.fx import Node


def _resolve_aten(target: object) -> "torch._ops.OpOverload | None":
    """Return the underlying aten OpOverload for a node target, or None.

    Handles both edge-dialect ops (EdgeOpOverload, whose ``_op`` is the aten op)
    and plain aten OpOverloads (whose ``_op`` is the C++ builtin, so it must not
    be unwrapped). See graph_serialize._resolve_op_overload for the same logic.
    """
    inner = getattr(target, "_op", None)
    if isinstance(inner, torch._ops.OpOverload):
        return inner
    if isinstance(target, torch._ops.OpOverload):
        return target
    return None


def _single_user(node: object) -> bool:
    return isinstance(node, Node) and len(node.users) == 1


def _fake_tensor(node: object) -> "torch.Tensor | None":
    val = node.meta.get("val") if isinstance(node, Node) else None
    return val if isinstance(val, torch.Tensor) else None
