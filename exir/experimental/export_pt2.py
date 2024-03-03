# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# This class is for prototyping PyTorch 2.0 Export
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple

import torch

from executorch import exir
from executorch.exir import CaptureConfig
from executorch.exir.error import ExportError, ExportErrorType, InternalError

from executorch.exir.tracer import Value

from torch._dynamo.guards import Guard as DynamoGuard


class GuardType(Enum):
    TENSOR_MATCH = 1


class GuardResolution(Enum):
    IGNORE = 1
    CHECK_AT_RUNTIME = 2
    ERROR_AT_EXPORT = 3


@dataclass
class Guard:
    """
    This is our own custom Guard class to store
    information needed for EXIR. This will only
    store things we actually need.
    """

    guard_type: GuardType
    obj: Any  # pyre-ignore
    check_code: str


@dataclass
class Trace:
    """
    Immutable object that abstracts the result of exir.trace
    which is essentially a torch.fx.GraphModule plus all the assumptions
    that are made about this tracing that are represented as Guard.
    """

    graph_module: torch.fx.GraphModule
    guards: List[Guard]
    inputs: Tuple[Value]


class ExportSession:
    def __init__(self, trace: Trace) -> None:
        """
        Mutable object where user can interactively resolve guards to access the final graph_module.
        """
        self.trace = trace
        self.guard_rules: List[Callable[[Guard], Optional[GuardResolution]]] = []

        # TODO (make more specific rule)
        def default_rule(guard: Guard) -> Optional[GuardResolution]:
            if guard.guard_type != GuardType.TENSOR_MATCH:
                return GuardResolution.IGNORE
            return None

        self.guard_rules.append(default_rule)

    def summary(self) -> str:
        """
        Prints the current status of guard resolutions in a module
        hierarchical way.
        """
        # TODO implement this
        return ""

    def export(self) -> Optional[torch.fx.GraphModule]:
        """
        Exports a final GraphModule that is ready to be executed.
        This will require that all guards imposed on GraphModule are
        resolved.
        """

        def _guard_remaining_filter(guard: Guard) -> bool:
            guard_resolutions: List[Optional[GuardResolution]] = [
                guard_rule(guard) for guard_rule in self.guard_rules
            ]
            # if there was no guard resolutions, we should keep the guard
            if len(guard_resolutions) == 0:
                return True

            # later rules take priority
            for idx in range(len(guard_resolutions) - 1, -1, -1):
                if guard_resolutions[idx] is None:
                    continue
                assert guard_resolutions is not None
                if guard_resolutions[idx] in [
                    GuardResolution.CHECK_AT_RUNTIME,
                    GuardResolution.IGNORE,
                ]:
                    return False
                if guard_resolutions[idx] == GuardResolution.ERROR_AT_EXPORT:
                    return True
            # nothing has been resolved
            return True

        remaining_guards = list(filter(_guard_remaining_filter, self.trace.guards))
        if len(remaining_guards) > 0:
            raise ExportError(
                ExportErrorType.VIOLATION_OF_SPEC,
                "There are outstanding guards to be resolved to export this graph",
            )
        return self.trace.graph_module

    def add_guard_rule(
        self, guard_rule: Callable[[Guard], Optional[GuardResolution]]
    ) -> None:
        """
        Adds user provided guard rule. This rule will be applied when you call export() method.
        """
        self.guard_rules.append(guard_rule)


def trace(root: Callable[..., Value], concrete_args: Tuple[Value, ...]) -> Trace:
    """
    Runs torchdynamo with no-python mode and dispatch trace
    to create a Trace object which is graph module plus guards that
    need to be resolved.
    """
    # TODO (yidi) cannot enable functionalization under exir.capture() pt2 mode
    graph_module = exir.capture(
        root,
        concrete_args,
        CaptureConfig(enable_functionalization=False),
    ).graph_module

    # TODO convert torchdynamo guards to our own guards
    def _convert_dynamo_guard_to_exir_guard(
        dynamo_guard: DynamoGuard,
    ) -> Optional[Guard]:
        if dynamo_guard.guard_types is not None and len(dynamo_guard.guard_types) > 0:
            # TODO (make sure this list is always element of 1)
            guard_type = dynamo_guard.guard_types[0]
            # TODO (add more guard types)
            if guard_type == "TENSOR_MATCH":
                # pyre-fixme[29]: `Optional[object]` is not a function.
                return Guard(GuardType.TENSOR_MATCH, dynamo_guard.obj_weakref(), "")

        raise InternalError(f"Unregistered guard type: {dynamo_guard.guard_types}")

    guards: List[Guard] = []
    for g in graph_module.guards:
        try:
            guard = _convert_dynamo_guard_to_exir_guard(g)
            assert isinstance(guard, Guard)
            guards.append(guard)
        except InternalError as e:
            print(str(e))

    return Trace(graph_module, guards, concrete_args)
