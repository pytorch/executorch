# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Preflight plans and the transactional rewrite of the Triton replacement pass.

An op whose calls a Triton kernel takes only sometimes registers a planner in
``replacement_pass.PLANNERS``. The planner is a pure preflight: it inspects a
node, never changes the graph, and classifies it:

- ``OPTIONAL``: a Triton kernel runs it and the regular lowering could too. A
  failed rewrite is rolled back and the node keeps the regular lowering.
- ``REQUIRED``: a Triton kernel runs it and the regular lowering cannot. A
  failure raises ``ReplacementError`` with the call's signature.
- ``UNSUPPORTED``: the node stays as it is; ``reason`` says why.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.fx import Graph, Node

logger: logging.Logger = logging.getLogger(__name__)


class Route(enum.Enum):
    OPTIONAL = "optional"
    REQUIRED = "required"
    UNSUPPORTED = "unsupported"


class ReplacementError(RuntimeError):
    """A call only a Triton kernel can compile could not be replaced."""


@dataclass(frozen=True)
class ReplacementPlan:
    route: Route
    reason: str
    # The kind of op ("conv") and the kernel chosen, for the pass's logs.
    kind: str = ""
    label: str = ""
    kernel: Any = None
    args: tuple[Any, ...] = ()
    # The normalized call, for errors.
    signature: str = ""


def device_supports(min_capability: int, device: torch.device) -> bool:
    """Whether a kernel that needs compute capability ``min_capability`` (e.g.
    80 for sm_80; 0 for every NVIDIA GPU) runs on ``device``, the GPU the model
    is lowered and autotuned on. Newer GPUs run it through the package's PTX;
    older ones load a variant exported for them."""
    if torch.version.hip is not None or device.type != "cuda":
        return False
    if min_capability == 0:
        return True
    major, minor = torch.cuda.get_device_capability(device)
    return 10 * major + minor >= min_capability


def unsupported(kind: str, reason: str) -> ReplacementPlan:
    return ReplacementPlan(Route.UNSUPPORTED, reason, kind)


def replace(graph: Graph, node: Node, plan: ReplacementPlan) -> bool:
    """Replaces ``node`` with ``plan.kernel(*plan.args)`` as one transaction.

    Every user's arguments are saved before they are rewired, so a failure at
    any step restores them all and removes the new node; a REQUIRED plan then
    raises, an OPTIONAL one logs and returns False.
    """
    saved = {user: (user.args, user.kwargs) for user in node.users}
    new_node: Optional[Node] = None
    try:
        with graph.inserting_before(node):
            new_node = graph.call_function(plan.kernel, plan.args)
        new_node.meta = node.meta.copy()
        for user in saved:
            user.replace_input_with(node, new_node)
        graph.erase_node(node)
    except Exception as e:
        for user, (args, kwargs) in saved.items():
            user.args, user.kwargs = args, kwargs
        if new_node is not None:
            graph.erase_node(new_node)
        if plan.route is Route.REQUIRED:
            raise ReplacementError(
                f"{plan.signature} ({plan.reason}): replacing {node.name} failed: {e!r}"
            ) from e
        logger.warning("Kept %s: replacing it failed: %r", node.name, e)
        return False
    return True
