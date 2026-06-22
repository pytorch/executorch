#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Type, TYPE_CHECKING, Union

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
        self._patterns: Dict[str, Type[PatternHandler]] = {}

    def reset(self) -> None:
        """Reset the registry to empty state. Useful for testing."""
        self._handlers.clear()
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
        t = node.target
        if t in self._handlers:
            return self._handlers[t]
        # Handle EdgeOpOverload by extracting the underlying ATen op
        if hasattr(t, "_op") and t._op in self._handlers:
            return self._handlers[t._op]
        # Check for string-based targets (e.g., higher_order ops)
        target_str = str(t)
        if target_str in self._handlers:
            return self._handlers[target_str]
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
