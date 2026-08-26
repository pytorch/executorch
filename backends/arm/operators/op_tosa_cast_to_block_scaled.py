# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Provide a visitor for lowering block-scaled casts to TOSA."""

import operator
from typing import Any, List

import torch
import tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_valid_dtype,
)
from executorch.backends.arm.tosa.mapping import TosaArg, TosaSpecialDtype
from executorch.backends.arm.tosa.specification import TosaSpecification


def _getitem_index(node: torch.fx.Node) -> int:
    index = node.args[1]
    if not isinstance(index, int):
        raise ValueError(
            f"CAST_TO_BLOCK_SCALED: expected integer getitem index, got {index!r}"
        )
    return index


def _ordered_getitem_outputs(node: torch.fx.Node) -> list[torch.fx.Node]:
    getitem_users = [
        user
        for user in node.users
        if user.op == "call_function" and user.target == operator.getitem
    ]

    ordered_users = sorted(getitem_users, key=_getitem_index)
    if len(ordered_users) != 2:
        raise ValueError(
            f"{CastToBlockScaledVisitor.target}: Expected exactly two getitem "
            f"outputs, got {len(ordered_users)}"
        )

    return ordered_users


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
        block_size = inputs[1].number if hasattr(inputs[1], "number") else None
        if not isinstance(block_size, int) or isinstance(block_size, bool):
            raise ValueError(f"{self.target}: missing block_size argument")

        validate_valid_dtype(
            self.target,
            input_tensor,
            [ts.DType.FP32, ts.DType.BF16, ts.DType.FP16],
            self.tosa_spec,
        )

        if not isinstance(node.meta.get("val"), tuple) or len(node.meta["val"]) != 2:
            raise ValueError(
                f"{self.target}: expected tuple metadata with two outputs, got {node.meta.get('val')!r}"
            )
        output_data_tensor, output_scale_tensor = node.meta["val"]
        output_getitems = _ordered_getitem_outputs(node)
        output_names = [user.name for user in output_getitems]
        output_payload_dtype = output_getitems[0].meta.get(TosaSpecialDtype.meta_key())

        if output_payload_dtype in (
            TosaSpecialDtype.FP4E2M1,
            TosaSpecialDtype.FP6E2M3,
            TosaSpecialDtype.FP6E3M2,
        ):
            output_data_dtype = output_payload_dtype.get_tosa_dtype()
        elif output_data_tensor.dtype == torch.float8_e4m3fn:
            output_data_dtype = ts.DType.FP8E4M3
        elif output_data_tensor.dtype == torch.float8_e5m2:
            output_data_dtype = ts.DType.FP8E5M2
        else:
            raise ValueError(
                f"{self.target}: unsupported payload dtype {output_data_tensor.dtype}"
            )
        if output_data_dtype not in (
            ts.DType.FP4E2M1,
            ts.DType.FP6E2M3,
            ts.DType.FP6E3M2,
            ts.DType.FP8E4M3,
            ts.DType.FP8E5M2,
        ):
            raise ValueError(
                f"{self.target}: unsupported payload dtype {output_data_dtype}"
            )
        if output_scale_tensor.dtype != torch.float8_e8m0fnu:
            raise ValueError(
                f"{self.target}: unsupported scale dtype {output_scale_tensor.dtype}"
            )

        if not hasattr(ts.Op, "CAST_TO_BLOCK_SCALED"):
            raise NotImplementedError(
                "tosa_serializer does not provide CAST_TO_BLOCK_SCALED yet"
            )

        attr = ts.TosaSerializerAttribute()
        attr_ctor = getattr(attr, "CastToBlockScaledAttribute", None)
        if attr_ctor is None:
            raise NotImplementedError(
                "tosa_serializer does not provide CastToBlockScaledAttribute yet"
            )
        attr_ctor(block_size)

        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.CAST_TO_BLOCK_SCALED,
            [input_tensor.name],
            output_names,
            attr,
        )
