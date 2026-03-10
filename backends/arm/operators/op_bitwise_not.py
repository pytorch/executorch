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
from executorch.backends.arm.tosa import TosaSpecification

INT_SPECS = TosaSpecification.all_versions_for_profile("INT")


@register_node_visitor
class BitwiseNotVisitor(SimpleNodeVisitor):
    target = "aten.bitwise_not.default"
    tosa_specs = INT_SPECS

    @classmethod
    def get_config(cls) -> SimpleNodeVisitorConfig:
        return SimpleNodeVisitorConfig(
            tosa_op=ts.Op.BITWISE_NOT,
            attr_method="BitwiseNotAttribute",
            num_inputs=1,
            input_dtypes=[ts.DType.INT8, ts.DType.INT16, ts.DType.INT32],
        )
