# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List

import torch
import tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa.mapping import TosaArg


@register_node_visitor
class CustomVisitor(NodeVisitor):
    """Lower the TOSA CUSTOM op from the TOSA backend dialect."""

    target = "tosa.CUSTOM.default"

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        allowed_kwargs = {"operator_name", "domain_name", "implementation_attrs"}
        unexpected = set(node.kwargs.keys()) - allowed_kwargs
        if unexpected:
            raise ValueError(
                f"tosa.CUSTOM received unexpected kwargs: {sorted(unexpected)}"
            )

        operator_name = node.kwargs.get("operator_name")
        domain_name = node.kwargs.get("domain_name")
        implementation_attrs = node.kwargs.get("implementation_attrs")

        if operator_name is None or domain_name is None:
            raise ValueError(
                "tosa.CUSTOM requires operator_name and domain_name in kwargs"
            )

        if implementation_attrs is None:
            impl_list = []
        elif isinstance(implementation_attrs, list):
            # NOTE: PyTorch schemas do not support a bytes type; we pass
            # implementation_attrs as int[] representing raw bytes.
            impl_list = [int(x) for x in implementation_attrs]
        else:
            raise TypeError(
                "implementation_attrs must be None or list[int]; "
                f"got {type(implementation_attrs)}"
            )

        attr = ts.TosaSerializerAttribute()
        attr.CustomAttribute(
            operator_name=operator_name,
            domain_name=domain_name,
            implementation_attrs=impl_list,
        )

        expanded = [TosaArg(item, self.tosa_spec) for item in inputs[0].special]
        input_names = [arg.name for arg in expanded]
        output_names = (
            output.multiple_output_names
            if getattr(output, "multiple_output_names", None)
            else [output.name]
        )
        if len(output_names) != 1:
            # TODO: Support multi-output CUSTOM ops with per-output meta/shape.
            raise ValueError(
                f"tosa.CUSTOM currently requires a single output, got {len(output_names)}"
            )
        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.CUSTOM,
            input_names,
            output_names,
            attr,
        )
