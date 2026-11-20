# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Preflight plans and the transactional rewrite of the Triton replacement pass.

An op whose calls a Triton kernel takes only sometimes registers a planner in
``replacement_pass.PLANNERS``. The planner is a pure preflight: it inspects a
node, never changes the graph, and lists the implementations that could run
it, each with the reason it cannot (None if it can): the regular lowering
(Inductor) and the family's replacement. The replacement pass then

- fails the export, with every reason, if none can;
- uses the only one that can;
- otherwise lets measured routing pick (see ``measured_routing.py``) for a
  family that opts in with ``measure``, or takes its ``static_choice``.

A failed rewrite to a replacement is rolled back and keeps the node if the
regular lowering can run it, and raises ``ReplacementError`` otherwise.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.fx import Graph, Node

logger: logging.Logger = logging.getLogger(__name__)

REGULAR_LOWERING = "inductor"


class ReplacementError(RuntimeError):
    """A call no other implementation can run could not be replaced."""


@dataclass(frozen=True)
class Candidate:
    """One implementation of a call: the regular lowering, or a replacement
    ``kernel(*args, **kwargs)``."""

    name: str
    # Why it cannot run the call; None if it can.
    reason: Optional[str] = None
    # For the pass's logs, e.g. the template.
    label: str = ""
    kernel: Any = None
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class ReplacementPlan:
    # The kind of op ("conv"), for logs and errors.
    kind: str
    # The regular lowering, then the replacement.
    candidates: tuple[Candidate, ...]
    # The family's static pick among the legal candidates, and why.
    static_choice: str = REGULAR_LOWERING
    static_reason: str = ""
    # The normalized call, for errors.
    signature: str = ""
    # Whether measured routing may pick for this family; otherwise its static
    # choice is taken.
    measure: bool = False

    def __post_init__(self) -> None:
        names = [c.name for c in self.candidates]
        if len(names) != 2 or names[0] != REGULAR_LOWERING or names[1] == names[0]:
            raise ReplacementError(
                f"{self.signature}: a plan lists the regular lowering and one "
                f"replacement, not {names}"
            )
        replacement = self.candidates[1]
        if replacement.reason is None and not (
            callable(replacement.kernel) and (replacement.args or replacement.kwargs)
        ):
            raise ReplacementError(
                f"{self.signature}: the replacement {replacement.name!r} can run "
                "the call but has no kernel and arguments to run it with"
            )
        legal = [c.name for c in self.legal()]
        if legal and self.static_choice not in legal:
            raise ReplacementError(
                f"{self.signature}: the static choice {self.static_choice!r} "
                f"cannot run the call ({self.describe()})"
            )

    def legal(self) -> list[Candidate]:
        return [c for c in self.candidates if c.reason is None]

    def candidate(self, name: str) -> Candidate:
        return next(c for c in self.candidates if c.name == name)

    def regular_lowering_runs_it(self) -> bool:
        return self.candidate(REGULAR_LOWERING).reason is None

    def describe(self) -> str:
        return "; ".join(f"{c.name}: {c.reason or 'legal'}" for c in self.candidates)


def regular_only(kind: str, reason: str) -> ReplacementPlan:
    """A plan whose replacement cannot run the call, for ``reason``."""
    return ReplacementPlan(kind, (Candidate(REGULAR_LOWERING), Candidate(kind, reason)))


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


def replace(
    graph: Graph, node: Node, plan: ReplacementPlan, choice: Candidate
) -> Optional[Node]:
    """Replaces ``node`` with ``choice.kernel(*choice.args, **choice.kwargs)``
    as one transaction, and returns the new node.

    Every user's arguments are saved before they are rewired, so a failure at
    any step restores them all and removes the new node; then, if the regular
    lowering can run the call, it logs and returns None, and otherwise
    raises.
    """
    saved = {user: (user.args, user.kwargs) for user in node.users}
    new_node: Optional[Node] = None
    try:
        with graph.inserting_before(node):
            new_node = graph.call_function(choice.kernel, choice.args, choice.kwargs)
        new_node.meta = node.meta.copy()
        for user in saved:
            user.replace_input_with(node, new_node)
        graph.erase_node(node)
    except Exception as e:
        for user, (args, kwargs) in saved.items():
            user.args, user.kwargs = args, kwargs
        if new_node is not None:
            graph.erase_node(new_node)
        if not plan.regular_lowering_runs_it():
            raise ReplacementError(
                f"{plan.signature} ({plan.describe()}): replacing {node.name} "
                f"failed: {e!r}"
            ) from e
        logger.warning("Kept %s: replacing it failed: %r", node.name, e)
        return None
    return new_node
