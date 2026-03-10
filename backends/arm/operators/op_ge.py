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

COMPARE_INPUT_DTYPES = [ts.DType.INT32, ts.DType.FP32, ts.DType.BF16, ts.DType.FP16]


@register_node_visitor
class GreaterEqualVisitor(SimpleNodeVisitor):
    target = "aten.ge.Tensor"

    @classmethod
    def get_config(cls) -> SimpleNodeVisitorConfig:
        return SimpleNodeVisitorConfig(
            tosa_op=ts.Op.GREATER_EQUAL,
            attr_method="GreaterEqualAttribute",
            num_inputs=2,
            input_dtypes=COMPARE_INPUT_DTYPES,
            output_dtypes=[ts.DType.BOOL],
            same_dtype_with_output=False,
            dtype_check_inputs_only=True,
        )
