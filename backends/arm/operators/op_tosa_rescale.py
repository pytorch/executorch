# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, List

import tosa_serializer as ts
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)

from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
)
from executorch.backends.arm.tosa.mapping import TosaArg
from torch.fx import Node


def _create_const_ops_for_rescale(
    tosa_fb,
    scale_32,
    input_dtype,
    node_name,
    multipliers,
    shifts,
    input_zp,
    output_zp,
    output_dtype,
    ts,
):
    """Materialize constant operands required by the TOSA RESCALE op.

    For TOSA spec v1.0 RESCALE operator requires multiplier, shifts, input_zp
    and output_zp to be const inputs. Create constant operators from the data
    already initialized.

    Args:
        tosa_fb (Any): Graph builder used to emit TOSA operators and tensors.
        scale_32 (bool): Flag indicating whether multipliers use 32-bit width.
        input_dtype (ts.DType): Data type of the input tensor.
        node_name (str): Base name reused for created constant tensors.
        multipliers (list[int]): Precomputed multiplier coefficients.
        shifts (list[int]): Precomputed shift coefficients.
        input_zp (list[int]): Quantization zero points for the input.
        output_zp (list[int]): Quantization zero points for the output.
        output_dtype (ts.DType): Data type of the output tensor.
        ts (module): Reference to the ``tosa_serializer`` module.

    Returns:
        list[str]: Names of the constant tensors added to ``tosa_fb`` in the
            order expected by RESCALE.

    """

    multipliers = tosa_fb.addConst(
        (len(multipliers),),
        ts.DType.INT32 if scale_32 else ts.DType.INT16,
        multipliers,
        name=node_name + "_multipliers",
    )
    shifts = tosa_fb.addConst(
        (len(shifts),), ts.DType.INT8, shifts, name=node_name + "_shifts"
    )
    input_zp = tosa_fb.addConst(
        [1], input_dtype, input_zp, name=node_name + "_input_zp"
    )
    output_zp = tosa_fb.addConst(
        [1], output_dtype, output_zp, name=node_name + "_output_zp"
    )

    return [multipliers.name, shifts.name, input_zp.name, output_zp.name]


@register_node_visitor
class RescaleVisitor(NodeVisitor):
    target = "tosa.RESCALE.default"

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        validate_num_inputs(self.target, inputs, 5)

        multiplier, shift, input_zp, output_zp = node.args[1:]
        scale32 = node.kwargs.get("scale32", True)
        rescale_inputs = _create_const_ops_for_rescale(
            tosa_graph,
            scale32,
            inputs[0].dtype,
            output.name,
            multiplier,
            shift,
            [input_zp],
            [output_zp],
            output.dtype,
            ts,
        )

        attr_rescale = ts.TosaSerializerAttribute()
        attr_rescale.RescaleAttribute(
            scale32=scale32,
            rounding_mode=ts.RoundingMode.__members__[
                node.kwargs.get("rounding_mode", "SINGLE_ROUND")
            ],
            per_channel=node.kwargs.get("per_channel", False),
            input_unsigned=node.kwargs.get("input_unsigned", False),
            output_unsigned=node.kwargs.get("output_unsigned", False),
        )

        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.RESCALE,
            [inputs[0].name, *rescale_inputs],
            [output.name],
            attr_rescale,
        )
