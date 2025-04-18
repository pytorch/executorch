# Copyright (c) 2024 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.fx import Node
from torch.nn import Parameter

from executorch.backends.nxp.backend.ir.converter.node_converter import NodeConverter, Target
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import BuiltinOperator


class ReLUConverter(NodeConverter):
    supported_targets = [Target.RT700]

    @staticmethod
    def _is_supported_in_IR(node: Node, parameters_mapping: dict[str, Parameter]) -> bool:
        return True

    def convert(self, node: Node):
        t_op = self._create_tflite_op_with_io_tensors(node)
        t_op.opcode_index = self.builder.op_code_index_for_op_type(BuiltinOperator.RELU)

        self.builder.append_operators([t_op])
