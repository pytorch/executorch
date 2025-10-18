# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json

from dataclasses import asdict, dataclass
from typing import Any, Optional

import serializer.tosa_serializer as ts  # type: ignore
import torch

from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec

from torch.fx.traceback import NodeSource


@dataclass
class TosaDebugSchema:
    node_name: str
    operator_name: str
    operator_id: int


@dataclass
class ATenDebugSchema:
    node_name: str
    operator_name: str

    @staticmethod
    def from_node(node: torch.fx.Node) -> ATenDebugSchema:
        # node.target is Union[Callable[..., Any], str], so we need to access this correctly depending on the type
        if callable(node.target):
            operator_name = node.target.__name__
        else:
            operator_name = node.target

        return ATenDebugSchema(node_name=node.name, operator_name=operator_name)


@dataclass
class TorchDebugSchema:
    stack_trace: list[str]
    node_trace: list[dict[str, Any]] | str
    nn_module_stack: dict[str, Any] | str
    torch_fn: tuple[str, str] | str

    @staticmethod
    def serialize_node_trace(node_trace: list[NodeSource]) -> list[dict[str, Any]]:
        """Flatten the from_node dictionary to remove nesting."""
        flattened = []
        node_stack = []

        for n in node_trace:
            node_stack.append((n, -1))

        while len(node_stack) > 0:
            node, parent_id = node_stack.pop()
            flattened.append(
                {
                    "name": node.name,
                    "target": node.target,
                    "graph_id": node.graph_id,
                    "pass_name": node.pass_name,
                    "action": node._get_action_string(),
                    "parent_graph_id": parent_id,
                }
            )

            for n in node.from_node:
                node_stack.append((n, node.graph_id))

        return flattened

    @staticmethod
    def from_node(node: torch.fx.Node) -> TorchDebugSchema:
        node_trace: str | list[dict[str, Any]] = "No node trace available."

        if "from_node" in node.meta:
            # Flatten the node_trace dictionary, so there is no nesting
            node_trace = TorchDebugSchema.serialize_node_trace(node.meta["from_node"])

        return TorchDebugSchema(
            stack_trace=node.meta.get("stack_trace", "No stack trace available").split(
                "\n"
            ),
            node_trace=node_trace,
            nn_module_stack=node.meta.get(
                "nn_module_stack", "No module stack trace available"
            ),
            torch_fn=node.meta.get("torch_fn", "No torch_fn available"),
        )


@dataclass
class DebugSchema:
    event_id: int
    aten_info: ATenDebugSchema
    tosa_info: Optional[TosaDebugSchema]
    torch_info: TorchDebugSchema

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)

        if self.tosa_info is None:
            output.pop("tosa_info")

        return output


class DebugHook:
    def __init__(self, debug_mode: ArmCompileSpec.DebugMode) -> None:
        self._debug_events: list[DebugSchema] = []
        self.__op_id_to_name = {}
        self.mode = debug_mode

        # Build up a mapping from TOSA 1.0 operator IDs to their names
        for name, val in vars(ts.Op).items():
            self.__op_id_to_name[val] = name

    def add(self, node: torch.fx.Node, tosa_op: Any, tosa_op_id: int) -> DebugSchema:
        tosa_debug_info = None

        # If the debug data is being embedded into the TOSA flatbuffer
        # do not collect TOSADebugSchema data, it's redundent
        if self.mode != ArmCompileSpec.DebugMode.TOSA:
            tosa_debug_info = TosaDebugSchema(
                node_name=str(tosa_op),
                operator_name=self.__op_id_to_name[tosa_op_id],
                operator_id=tosa_op_id,
            )

        aten_debug_info = ATenDebugSchema.from_node(node)
        torch_debug_info = TorchDebugSchema.from_node(node)

        debug_info = DebugSchema(
            event_id=len(self._debug_events),
            aten_info=aten_debug_info,
            tosa_info=tosa_debug_info,
            torch_info=torch_debug_info,
        )
        self._debug_events.append(debug_info)

        return debug_info

    def serialize(self) -> str:
        return json.dumps([event.to_dict() for event in self._debug_events], indent=4)
