# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from torch.fx import Node
from torch.nn import Parameter


class HardTanhConverter(NodeConverter):

    # Maps possible input parameters of HardTanh to equivalent ReLU-based operators supported by TFLite.
    supported_modes_map = {
        (0.0, 6.0): BuiltinOperator.RELU6,
        (-1.0, 1.0): BuiltinOperator.RELU_N1_TO_1,
        (0.0, 1.0): BuiltinOperator.RELU_0_TO_1,
        (0.0, float("inf")): BuiltinOperator.RELU,
    }

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        _, min_value, max_value = node.args
        return (min_value, max_value) in HardTanhConverter.supported_modes_map.keys()

    def convert(self, node: Node):
        """Convert 'aten::hardtanh' to it's supported ReLU equivalent."""
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)

        _, min_value, max_value = node.args

        op = self.supported_modes_map[(min_value, max_value)]
        t_op.opcode_index = self.builder.op_code_index_for_op_type(op)

        self.builder.append_operators([t_op])
