# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import tosa_serializer as ts
from executorch.backends.arm.operators.node_visitor import register_node_visitor
from executorch.backends.arm.operators.op_tosa_conv2d import Conv2dVisitor
from executorch.backends.arm.tosa import TosaSpecification


@register_node_visitor
class DepthwiseConv2dVisitor(Conv2dVisitor):
    """Provide a visitor that serializes TOSA ``DEPTHWISE_CONV2D``."""

    target = "tosa.DEPTHWISE_CONV2D.default"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

    def _get_tosa_op(self):
        return ts.Op.DEPTHWISE_CONV2D

    def _get_attr_func(self, attr):
        return attr.DepthwiseConv2dAttribute

    # Inheriting the define_node method from Conv2dVisitor
