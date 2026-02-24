# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared pattern matching utilities for MLX backend.

This module provides common utilities used by both:
- passes.py: Graph transformation passes (ExportPass)
- patterns.py: MLX lowering pattern handlers (PatternHandler)

The core abstraction is the `PatternMatch` base class which provides:
- `maybe_create(head)` - Class method to match a pattern from a head node
- Captured values as typed fields
- `body` list of intermediate nodes to remove

Usage in passes.py:
    class FuseRMSNormPass(ExportPass):
        def call(self, graph_module):
            for node in graph.nodes:
                if match := RMSNormMatch.maybe_create(node):
                    replacement = self._emit_fused_op(graph, match)
                    node.replace_all_uses_with(replacement)
                    match.remove_body_nodes(graph)

Usage in patterns.py:
    class RMSNormHandler(PatternHandler):
        @classmethod
        def maybe_create(cls, ep, head):
            if match := RMSNormMatch.maybe_create(head):
                return cls(head, match.body, match.input, match.weight, match.eps)
            return None
"""

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Set, Tuple, Union

from executorch.backends.mlx.program_builder import get_aten_target_normalized
from torch.fx import Graph
from torch.fx.node import Node


# Type alias for walk_back result entries
# Each entry corresponds to an OpStep:
#   - Node: matched node (for regular steps)
#   - None: optional step that didn't match
#   - List[Node]: repeat step (0 or more matches)
WalkBackEntry = Union[Node, None, List[Node]]


# =============================================================================
# Node Matching Utilities
# =============================================================================


def match_target(node: Node, op: Any) -> bool:
    """
    Check if a node's normalized aten target matches the given op.

    Uses get_aten_target_normalized to handle edge dialect ops.
    This means slice_copy matches slice, etc.

    Args:
        node: The node to check
        op: The op to match (e.g., torch.ops.aten.mul.Tensor)
    """
    return node.op == "call_function" and get_aten_target_normalized(node.target) == op


def has_single_user(node: Node) -> bool:
    """Check if a node has exactly one consumer."""
    return len(node.users) == 1


def has_no_users(node: Node) -> bool:
    """Check if a node has no consumers (dead code)."""
    return len(node.users) == 0


def extract_lifted_tensor_constant(node: Node) -> Optional[float]:
    """
    Extract scalar value from a lifted tensor constant node.

    Lifted constants are created during torch.export and contain small
    constant tensors (like epsilon values). The actual value is stored
    in node.meta["val"].

    Args:
        node: A node that may be a lifted tensor constant

    Returns:
        The scalar float value, or None if not a lifted constant or not scalar
    """
    if not isinstance(node, Node):
        return None
    if "lifted_tensor_constant" not in node.name:
        return None
    val = node.meta.get("val")
    if val is None:
        return None
    if not hasattr(val, "item"):
        return None
    try:
        return float(val.item())
    except (RuntimeError, ValueError):
        return None


# =============================================================================
# Pattern Walking Infrastructure
# =============================================================================


@dataclass
class OpStep:
    """
    One step in a backward walk through the graph.

    Used with walk_back() to define pattern chains. Supports both exact op
    matching and predicate-based matching.

    Attributes:
        op: Specific op to match (e.g., torch.ops.aten.rsqrt.default)
        predicate: Alternative to op - a function that returns True for matching nodes
        optional: If True, skip this step if it doesn't match
        repeat: If True, match this step 0 or more times (like regex *)
        require_single_user: If True (default), only match nodes with exactly one user
        nargs: Number of args required. Can be:
               - int: minimum number of args (default 1, since we advance via args[0])
               - tuple (min, max): range of args required (inclusive)
        kwargs: Set of kwargs we handle (node's kwargs must be subset of this)
        arg_index: Which arg to follow when advancing (default 0)

    Examples:
        # Match specific op
        OpStep(op=torch.ops.aten.rsqrt.default)

        # Match with predicate (for matching families of ops)
        OpStep(predicate=lambda n: match_target(n, torch.ops.aten.select.int))

        # Match chain of same op type (0 or more)
        OpStep(op=torch.ops.aten.select.int, repeat=True)

        # Optional dtype conversion
        OpStep(op=torch.ops.aten._to_copy.default, optional=True)

        # Require between 2 and 4 args
        OpStep(op=torch.ops.aten.some_op.default, nargs=(2, 4))

        # Declare that we handle 'dtype' kwarg
        OpStep(op=torch.ops.aten._to_copy.default, kwargs={"dtype"})

        # Follow second arg (e.g., mul(x, rsqrt(y)) -> follow rsqrt in args[1])
        OpStep(op=torch.ops.aten.mul.Tensor, arg_index=1)
    """

    op: Any = None
    predicate: Optional[Callable[[Node], bool]] = None
    optional: bool = False
    repeat: bool = False
    require_single_user: bool = True
    nargs: Union[int, Tuple[int, int]] = 1
    kwargs: Set[str] = field(default_factory=set)  # Empty = no kwargs allowed
    arg_index: int = 0

    def matches(self, node: Node) -> bool:
        """Check if this step fully matches the given node."""
        # Check op or predicate
        if self.op is not None:
            if not match_target(node, self.op):
                return False
        elif self.predicate is not None:
            if not self.predicate(node):
                return False
        else:
            return False

        # Check single user requirement
        if self.require_single_user and not has_single_user(node):
            return False

        # Check nargs and kwargs
        if not self._check_nargs(node):
            return False
        if not self._check_kwargs(node):
            return False

        return True

    def _check_nargs(self, node: Node) -> bool:
        """Check if node has the required number of args."""
        n = len(node.args)
        if isinstance(self.nargs, tuple):
            min_args, max_args = self.nargs
            # Must be in range AND enough to access arg_index
            return min_args <= n <= max_args and n > self.arg_index
        else:
            # Must have at least nargs, AND enough to access arg_index
            return n >= self.nargs and n > self.arg_index

    def _check_kwargs(self, node: Node) -> bool:
        """Check that node's kwargs are all declared in self.kwargs (no unhandled kwargs)."""
        return set(node.kwargs.keys()).issubset(self.kwargs)


def walk_back(  # noqa: C901
    node: Node,
    steps: List[OpStep],
    debug: bool = False,
) -> Optional[Tuple[Node, List[WalkBackEntry]]]:
    """
    Walk backwards through a chain of ops, matching against a pattern.

    Starting from *node*, try to match each step against the current node.
    At every matched step the walk advances to ``cur.args[step.arg_index]``.
    Optional steps are silently skipped when they don't match. Repeat steps
    match 0 or more times.

    Args:
        node: Starting node
        steps: List of OpStep to match in order

    Returns:
        ``(base_node, entries)`` if the full chain matches, else ``None``.
        *base_node* is the input to the first (deepest) op in the chain.
        *entries* is a list with one entry per OpStep:
            - Node: matched node (for regular steps)
            - None: optional step that didn't match
            - List[Node]: repeat step (0 or more matches)

    Examples:
        # Match: rsqrt(add(mean(pow(x, 2)), eps))
        result = walk_back(rsqrt_node, [
            OpStep(op=torch.ops.aten.rsqrt.default),
            OpStep(op=torch.ops.aten.add.Tensor),
            OpStep(op=torch.ops.aten.mean.dim),
            OpStep(op=torch.ops.aten.pow.Tensor_Scalar),
        ])
        if result:
            base, entries = result
            rsqrt, add, mean, pow = entries  # Each is a Node

        # Match chain of select ops (like tensor[0][0])
        result = walk_back(node, [
            OpStep(op=torch.ops.aten.select.int, repeat=True),
        ])
        if result:
            base, entries = result
            select_nodes = entries[0]  # List[Node], may be empty

        # Skip optional _to_copy, then match rsqrt
        result = walk_back(node, [
            OpStep(op=torch.ops.aten._to_copy.default, optional=True),
            OpStep(op=torch.ops.aten.rsqrt.default),
        ])
        if result:
            base, entries = result
            to_copy, rsqrt = entries  # to_copy may be None
    """
    entries: List[WalkBackEntry] = []
    cur = node

    for i, step in enumerate(steps):
        if not isinstance(cur, Node):
            if debug:
                print(
                    f"  [walk_back] step {i}: cur is not a Node ({type(cur).__name__})"
                )
            return None

        if step.repeat:
            # Match 0 or more times, return as list
            matched_nodes: List[Node] = []
            while isinstance(cur, Node) and step.matches(cur):
                matched_nodes.append(cur)
                cur = cur.args[step.arg_index]
            entries.append(matched_nodes)
            if debug:
                print(
                    f"  [walk_back] step {i} (repeat): matched {len(matched_nodes)} nodes"
                )
            # repeat always succeeds (matches 0 or more)
            continue

        if step.matches(cur):
            entries.append(cur)
            if debug:
                print(f"  [walk_back] step {i}: matched {cur.name}")
            cur = cur.args[step.arg_index]
        elif step.optional:
            entries.append(None)
            if debug:
                print(f"  [walk_back] step {i} (optional): skipped, cur={cur.name}")
            continue
        else:
            if debug:
                print(
                    f"  [walk_back] step {i}: FAILED at cur={cur.name}, target={cur.target}, step.op={step.op}"
                )
            return None

    if not isinstance(cur, Node):
        return None

    return cur, entries


# =============================================================================
# Pattern Match Base Class
# =============================================================================


@dataclass
class PatternMatch:
    """
    Base class for pattern match results.

    Subclasses should:
    1. Add fields for captured values (input nodes, constants, etc.)
    2. Implement maybe_create() classmethod for pattern matching
    3. Optionally implement emit_* methods for specific backends

    Example:
        @dataclass
        class RMSNormMatch(PatternMatch):
            input_node: Node
            weight_node: Node
            eps: float

            @classmethod
            def maybe_create(cls, head: Node) -> Optional["RMSNormMatch"]:
                # Pattern matching logic...
                if not matched:
                    return None
                return cls(
                    head=head,
                    body=body_nodes,
                    input_node=input_node,
                    weight_node=weight_node,
                    eps=eps_value,
                )
    """

    head: Node  # The output node of the matched pattern
    body: List[Node] = field(default_factory=list)  # Intermediate nodes

    @classmethod
    def maybe_create(cls, head: Node, **context) -> Optional["PatternMatch"]:
        """
        Try to match the pattern starting from head node.

        Override in subclasses to implement pattern-specific matching.

        Args:
            head: Candidate head node to match from
            **context: Additional context (e.g., ExportedProgram for patterns.py)

        Returns:
            PatternMatch instance with captured values, or None if no match
        """
        return None

    def remove_body_nodes(self, graph: Graph) -> None:
        """
        Remove body nodes from the graph (in reverse order for safety).

        Call after replacing head with fused op.
        """
        for node in reversed(self.body):
            if has_no_users(node):
                graph.erase_node(node)

    def all_nodes(self) -> List[Node]:
        """Return all nodes in the pattern (head + body)."""
        return [self.head] + self.body
