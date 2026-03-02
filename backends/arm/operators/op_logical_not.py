# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import register_node_visitor
from executorch.backends.arm.operators.simple_node_visitor import (
    SimpleNodeVisitor,
    SimpleNodeVisitorConfig,
)


@register_node_visitor
class LogicalNotVisitor(SimpleNodeVisitor):
    target = "aten.logical_not.default"

    @classmethod
    def get_config(cls) -> SimpleNodeVisitorConfig:
        return SimpleNodeVisitorConfig(
            tosa_op=ts.Op.LOGICAL_NOT,
            attr_method="LogicalNotAttribute",
            num_inputs=1,
            input_dtypes=[ts.DType.BOOL],
        )
