# Copyright 2023-2026 Arm Limited and/or its affiliates.
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
class SubVisitor(SimpleNodeVisitor):
    target = "aten.sub.Tensor"

    @classmethod
    def get_config(cls) -> SimpleNodeVisitorConfig:
        return SimpleNodeVisitorConfig(
            tosa_op=ts.Op.SUB,
            attr_method="SubAttribute",
            num_inputs=2,
            input_dtypes=[ts.DType.INT32, ts.DType.FP16, ts.DType.FP32, ts.DType.BF16],
        )
