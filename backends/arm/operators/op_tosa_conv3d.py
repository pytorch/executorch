# Copyright 2023-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Provide a visitor for lowering 3D convolution to TOSA (INT/FP)."""

from executorch.backends.arm.operators.node_visitor import register_node_visitor
from executorch.backends.arm.operators.op_tosa_conv2d import Conv2dVisitor


@register_node_visitor
class Conv3dVisitor(Conv2dVisitor):
    """Provide a visitor that serializes TOSA ``CONV3D``."""

    target = "tosa.CONV3D.default"

    def _get_tosa_op(self):
        import serializer.tosa_serializer as ts  # type: ignore

        return ts.Op.CONV3D

    def _get_attr_func(self, attr):
        return attr.Conv3dAttribute
