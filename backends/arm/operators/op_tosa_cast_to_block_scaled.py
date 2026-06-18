# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Provide a visitor for lowering block-scaled casts to TOSA."""

import operator
from typing import Any, cast, List

import torch
import tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
)
from executorch.backends.arm.tosa.mapping import TosaArg
from executorch.backends.arm.tosa.specification import TosaSpecification


def _ordered_getitem_output_names(node: torch.fx.Node) -> list[str]:
    getitem_users = [
        user
        for user in node.users
        if user.op == "call_function" and user.target == operator.getitem
    ]

    ordered_users = sorted(getitem_users, key=lambda user: cast(int, user.args[1]))
    if len(ordered_users) != 2:
        raise ValueError(
            f"{CastToBlockScaledVisitor.target}: Expected exactly two getitem outputs, got {len(ordered_users)}"
        )

    return [user.name for user in ordered_users]


@register_node_visitor
class CastToBlockScaledVisitor(NodeVisitor):
    """Serialize TOSA ``CAST_TO_BLOCK_SCALED``."""

    target = "tosa.CAST_TO_BLOCK_SCALED.default"
    tosa_specs = [TosaSpecification.create_from_string("TOSA-1.1+FP")]

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        validate_num_inputs(self.target, inputs, 2)
        # The tosa_specs attribute cannot express extension requirements.
        # Therefore, check for the extension explicitly here.
        if not self.tosa_spec.support_extension("mxfp"):
            raise ValueError(f"{self.target} requires the TOSA mxfp extension")

        input_tensor = inputs[0]
        block_size = inputs[1].number
        output_data_tensor, output_scale_tensor = node.meta["val"]

        # TODO(MLETORCH-2018): This is a local workaround for multi-output TOSA ops.
        # Remove it once twe can handle multiple outputs generally.
        output_names = _ordered_getitem_output_names(node)

        attr = ts.TosaSerializerAttribute()
        attr.CastToBlockScaledAttribute(block_size)

        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.CAST_TO_BLOCK_SCALED,
            [input_tensor.name],
            output_names,
            attr,
        )
