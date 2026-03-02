# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Shared base class for TOSA operator visitors."""

from dataclasses import dataclass
from typing import Any, List

import torch.fx
import tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import NodeVisitor
from executorch.backends.arm.tosa.mapping import TosaArg


@dataclass(frozen=True)
class SimpleNodeVisitorConfig:
    """Configuration bundle for ``SimpleNodeVisitor``."""

    tosa_op: ts.Op
    attr_method: str
    num_inputs: int | List[int]
    input_dtypes: List[Any]
    attr_kwargs: dict[str, Any] | None = None


class SimpleNodeVisitor(NodeVisitor):
    """Provide shared validation and emit helpers for TOSA visitors."""

    target: str

    @classmethod
    def get_config(cls) -> SimpleNodeVisitorConfig:
        raise NotImplementedError(
            f"{cls.__name__} must implement get_config or define define_node."
        )

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        cfg = self.get_config()

        self.validate(
            target=self.target,
            inputs=inputs,
            output=output,
            num_inputs=cfg.num_inputs,
            input_dtypes=cfg.input_dtypes,
        )

        self.serialize(
            node,
            tosa_graph,
            tosa_op=cfg.tosa_op,
            inputs=inputs,
            output=output,
            attr_method=cfg.attr_method,
            attr_kwargs=cfg.attr_kwargs,
        )
