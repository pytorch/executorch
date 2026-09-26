#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

from executorch.backends.mlx._logging import logger
from torch.fx.node import Node

if TYPE_CHECKING:
    from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
    from executorch.backends.mlx.builder.slot_manager import Slot
    from torch.export import ExportedProgram

# Handler type: takes (builder, node) and returns optional slot(s)
Handler = Callable[
    ["MLXProgramBuilder", Node], Optional[Union["Slot", Tuple["Slot", ...]]]
]

# Support-check type: takes (builder, node) and returns whether the handler for
# that node will lower it. See PatternHandler.supported for the contract.
SupportCheck = Callable[["MLXProgramBuilder", Node], bool]


class PatternHandler:
    def __init__(self, head: Node, body: List[Node]) -> None:
        self.head: Node = head
        self.body: List[Node] = body

    @classmethod
    def deferred_handler(cls, P: MLXProgramBuilder, n: Node) -> None:
        pass

    @classmethod
    def maybe_create(cls, ep: ExportedProgram, head: Node) -> Optional[PatternHandler]:
        raise NotImplementedError

    def __call__(self, P: MLXProgramBuilder, n: Node) -> None:
        raise NotImplementedError

    def supported(self, P: MLXProgramBuilder, n: Node) -> bool:
        """Optional: answer "would __call__ lower this node?" without running it.

        Support is normally decided by running the handler and seeing whether it
        throws, but handlers that repack a quantized weight copy the whole tensor
        to answer that question -- and the partitioner asks it several times per
        export. Overriding this lets such a handler answer from node metadata
        instead; it is then consulted in place of running the handler during
        support checks (never during build(), which has to emit).

        Implementations must decide from ``n.meta`` alone and must never read
        constant data, which is the cost being avoided. They must also agree with
        __call__: saying yes to a node the handler then rejects makes
        ops_to_not_decompose preserve an op that afterwards neither decomposes
        nor lowers. Disagreement is caught by the op tests rather than a
        dedicated check -- a false positive fails the export outright when
        build() runs the handler for real, and a false negative shows up as a
        missing delegate segment.
        """
        raise NotImplementedError

    @classmethod
    def has_support_check(cls) -> bool:
        """Whether this class overrides supported()."""
        return cls.supported is not PatternHandler.supported

    def set_handlers(self, P: MLXProgramBuilder):
        if P.node_info[self.head].handler is not None:
            raise AssertionError(
                f"Head node {self.head.name} already has handler {P.node_info[self.head].handler}, "
                f"cannot set pattern {self.__class__.__name__}"
            )
        for n in self.body:
            if P.node_info[n].handler is not None:
                raise AssertionError(
                    f"Body node {n.name} already has handler {P.node_info[n].handler}, "
                    f"cannot set pattern {self.__class__.__name__}"
                )

        logger.debug(
            f"Pattern {self.__class__.__name__}: "
            f"HEAD={self.head.name}, BODY={[n.name for n in self.body]}"
        )
        P.node_info[self.head].handler = self
        for n in self.body:
            P.node_info[n].handler = PatternHandler.deferred_handler


class MLXOpRegistry:
    """Registry for op handlers and pattern handlers."""

    def __init__(self):
        self._handlers: Dict[Union[str, Callable], Handler] = {}
        self._support_checks: Dict[Union[str, Callable], SupportCheck] = {}
        self._patterns: Dict[str, Type[PatternHandler]] = {}

    def reset(self) -> None:
        """Reset the registry to empty state. Useful for testing."""
        self._handlers.clear()
        self._support_checks.clear()
        self._patterns.clear()

    def register(self, target: Union[str, Callable, list, tuple]):
        """Decorator to register a handler for one or more op targets."""

        def deco(fn: Handler):
            targets = target if isinstance(target, (list, tuple)) else [target]
            for t in targets:
                if t in self._handlers:
                    raise ValueError(f"Target {t} already registered")
                self._handlers[t] = fn
            return fn

        return deco

    def get_handler(self, node: Node) -> Optional[Handler]:
        """Get the handler for a node, or None if not registered."""
        return self._lookup(self._handlers, node)

    def register_support_check(self, target: Union[str, Callable, list, tuple]):
        """Decorator registering a cheap support predicate for an op handler.

        The predicate takes (builder, node) and returns whether the handler will
        lower the node. It is consulted instead of running the handler during
        support checks, so it must decide from node metadata alone -- reading
        constant data is exactly the cost being avoided. See
        PatternHandler.supported for the full contract.
        """

        def deco(fn: SupportCheck):
            targets = target if isinstance(target, (list, tuple)) else [target]
            for t in targets:
                if t in self._support_checks:
                    raise ValueError(f"Support check for {t} already registered")
                self._support_checks[t] = fn
            return fn

        return deco

    def get_support_check(self, node: Node) -> Optional[SupportCheck]:
        """Get the support predicate for a node, or None if it has none."""
        return self._lookup(self._support_checks, node)

    @staticmethod
    def _lookup(table: Dict[Union[str, Callable], Any], node: Node) -> Optional[Any]:
        t = node.target
        if t in table:
            return table[t]
        # Handle EdgeOpOverload by extracting the underlying ATen op
        if hasattr(t, "_op") and t._op in table:
            return table[t._op]
        # Check for string-based targets (e.g., higher_order ops)
        target_str = str(t)
        if target_str in table:
            return table[target_str]
        return None

    def registered_ops(self) -> set:
        """Return all registered op targets."""
        return set(self._handlers.keys())

    def unregister(self, target: Union[str, Callable, list, tuple]) -> None:
        """Remove a handler for one or more op targets.

        This is useful for debugging - allows temporarily disabling specific
        handlers to test if they are causing issues.

        Args:
            target: Single target or list of targets to unregister
        """
        targets = target if isinstance(target, (list, tuple)) else [target]
        for t in targets:
            if t in self._handlers:
                del self._handlers[t]
            if t in self._support_checks:
                del self._support_checks[t]

    def register_pattern(self, name: str):
        """Decorator to register a pattern handler class."""

        def deco(cls: Type[PatternHandler]):
            if not issubclass(cls, PatternHandler):
                raise TypeError(
                    "register_pattern must decorate a PatternHandler subclass"
                )
            if name in self._patterns:
                raise ValueError(f"Pattern '{name}' already registered")
            self._patterns[name] = cls
            return cls

        return deco

    def get_pattern_cls(self, name: str) -> Optional[Type[PatternHandler]]:
        """Get a pattern handler class by name."""
        return self._patterns.get(name)

    def get_noop_handler(self) -> Optional[Handler]:
        """Get the NOOP handler, if registered."""
        return self._handlers.get("NOOP")

    def patterns(self):
        """Return all registered pattern names."""
        return self._patterns.keys()


# Global registry
REGISTRY = MLXOpRegistry()
