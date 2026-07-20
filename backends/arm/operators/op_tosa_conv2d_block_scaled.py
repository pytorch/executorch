# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Provide a visitor for lowering block-scaled Conv2d to TOSA."""

from typing import Any, cast, List

import tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_valid_dtype,
)
from executorch.backends.arm.tosa.mapping import TosaArg
from executorch.backends.arm.tosa.specification import TosaSpecification


def _build_conv2d_block_scaled_attr(
    attr: ts.TosaSerializerAttribute,
    *,
    block_size: int,
) -> None:
    attr_ctor = getattr(attr, "Conv2dBlockScaledAttribute", None)
    if attr_ctor is None:
        raise NotImplementedError(
            "tosa_serializer does not provide Conv2dBlockScaledAttribute yet"
        )

    attr_ctor(block_size)


@register_node_visitor
class Conv2dBlockScaledVisitor(NodeVisitor):
    """Serialize TOSA ``CONV2D_BLOCK_SCALED``."""

    target = "tosa.CONV2D_BLOCK_SCALED.default"
    tosa_specs = [TosaSpecification.create_from_string("TOSA-1.1+FP")]

    def define_node(
        self,
        node: Any,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        # The tosa_specs attribute cannot express extension requirements.
        # Therefore, check for the extension explicitly here.
        if not self.tosa_spec.support_extension("mxfp"):
            raise ValueError(f"{self.target} requires the TOSA mxfp extension")

        validate_num_inputs(self.target, inputs, 9)
        validate_valid_dtype(
            self.target,
            [inputs[0], inputs[2]],
            [
                ts.DType.FP4E2M1,
                ts.DType.FP6E2M3,
                ts.DType.FP6E3M2,
                ts.DType.FP8E4M3,
                ts.DType.FP8E5M2,
            ],
            self.tosa_spec,
        )
        validate_valid_dtype(
            self.target,
            [inputs[1], inputs[3]],
            ts.DType.FP8UE8M0,
            self.tosa_spec,
        )
        validate_valid_dtype(self.target, inputs[4], ts.DType.FP32, self.tosa_spec)
        validate_valid_dtype(
            self.target,
            [inputs[5], inputs[6], inputs[7]],
            ts.DType.SHAPE,
            self.tosa_spec,
        )
        validate_valid_dtype(self.target, output, ts.DType.FP32, self.tosa_spec)

        if not hasattr(ts.Op, "CONV2D_BLOCK_SCALED"):
            raise NotImplementedError(
                "tosa_serializer does not provide CONV2D_BLOCK_SCALED yet"
            )

        # TosaArg.number is float | int, but the fake-op schema guarantees int.
        block_size = cast(int, inputs[8].number)

        attr = ts.TosaSerializerAttribute()
        _build_conv2d_block_scaled_attr(
            attr,
            block_size=block_size,
        )

        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.CONV2D_BLOCK_SCALED,
            [
                inputs[0].name,
                inputs[1].name,
                inputs[2].name,
                inputs[3].name,
                inputs[4].name,
                inputs[6].name,
                inputs[5].name,
                inputs[7].name,
            ],
            [output.name],
            attr,
        )
