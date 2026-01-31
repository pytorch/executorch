# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, List

import torch
import torch.fx

import tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_same_dtype,
    validate_valid_dtype,
)
from executorch.backends.arm.tosa.mapping import TosaArg


def identity_operator_factory(identity_target: str):
    """
    Creates and registers NodeVisitors for operators that map directly
    to a TOSA IDENTITY op.
    """

    class IdentityOperatorVisitor(NodeVisitor):
        target = identity_target

        def define_node(
            self,
            node: torch.fx.Node,
            tosa_graph: Any,
            inputs: List[TosaArg],
            output: TosaArg,
        ) -> None:
            validate_num_inputs(self.target, inputs, 1)
            validate_same_dtype(self.target, [inputs[0], output], ts)
            supported_dtypes = [
                ts.DType.BOOL,
                ts.DType.INT8,
                ts.DType.INT16,
                ts.DType.INT32,
            ]
            if self.tosa_spec.support_float():
                supported_dtypes += [ts.DType.FP32]
            if self.tosa_spec.support_extension("bf16"):
                supported_dtypes += [ts.DType.BF16]
            if self.tosa_spec.support_extension("int16"):
                supported_dtypes += [ts.DType.INT48]
            if self.tosa_spec.support_extension("int4"):
                supported_dtypes += [ts.DType.INT4]
            validate_valid_dtype(
                self.target,
                [inputs[0], output],
                supported_dtypes,
                self.tosa_spec,
            )

            # Simply add an identityOp
            attr = ts.TosaSerializerAttribute()
            attr.IdentityAttribute()
            self._serialize_operator(
                node,
                tosa_graph,
                ts.Op.IDENTITY,
                [inputs[0].name],
                [output.name],
                attr,
            )

    register_node_visitor(IdentityOperatorVisitor)


identity_operator_factory("aten.alias_copy.default")
