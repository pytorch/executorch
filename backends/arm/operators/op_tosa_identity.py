# Copyright 2026 Arm Limited and/or its affiliates.
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
class IdentityVisitor(SimpleNodeVisitor):
    """Lower the TOSA IDENTITY op."""

    target = "tosa.IDENTITY.default"

    @classmethod
    def get_config(cls) -> SimpleNodeVisitorConfig:
        return SimpleNodeVisitorConfig(
            tosa_op=ts.Op.IDENTITY,
            attr_method="IdentityAttribute",
            num_inputs=1,
            input_dtypes=[
                ts.DType.BOOL,
                ts.DType.INT8,
                ts.DType.INT16,
                ts.DType.INT32,
                ts.DType.FP16,
                ts.DType.FP32,
                ts.DType.BF16,
                ts.DType.FP8E4M3,
                ts.DType.FP8E5M2,
            ],
        )
