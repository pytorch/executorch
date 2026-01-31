# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, List

import torch

import tosa_serializer as ts
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_valid_dtype,
)

from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.mapping import TosaArg


@register_node_visitor
class TableVisitor(NodeVisitor):
    target = "tosa.TABLE.default"

    tosa_specs = TosaSpecification.all_versions_for_profile("INT")

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        validate_num_inputs(self.target, inputs, 2)
        supported_input_dtypes = [ts.DType.INT8]
        supported_output_dtypes = [ts.DType.INT8]
        if self.tosa_spec.support_extension("int16"):
            supported_input_dtypes.append(ts.DType.INT16)
            supported_output_dtypes.append(ts.DType.INT32)

        validate_valid_dtype(
            self.target, inputs, supported_input_dtypes, self.tosa_spec
        )
        validate_valid_dtype(
            self.target, output, supported_output_dtypes, self.tosa_spec
        )

        # The name of the table constant is a bit complex.
        # The name of the pytorch buffer will be the target of last node argument.
        # However, when it is serialized to TOSA, a submodule suffix might be added. The TOSA buffer name thus
        # needs to be taken from the last TosaArg.
        pytorch_table_buffer_name = node.args[-1].target  # type: ignore[union-attr]
        tosa_table_buffer_name = inputs[-1].name
        if pytorch_table_buffer_name not in self._exported_program.state_dict.keys():
            raise RuntimeError(
                f"Did not find key {node.name} in state_dict {self._exported_program.state_dict.keys()}."
            )

        attr = ts.TosaSerializerAttribute()
        attr.TableAttribute()
        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.TABLE,
            [inputs[0].name, tosa_table_buffer_name],
            [output.name],
            attr,
        )
