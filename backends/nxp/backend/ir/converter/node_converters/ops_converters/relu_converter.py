# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    is_not_qdq_node,
    NodeConverter,
    Partition,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.neutron_operator_support import (
    activation_supported_on_target,
    NeutronTargetSpec,
)
from torch.fx import Node
from torch.nn import Parameter


class ReLUConverter(NodeConverter):

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        return True

    @classmethod
    def supports_partitioning_result(
        cls,
        node: Node,
        partition_list: list[Partition],
        custom_delegation_options: CustomDelegationOptions,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
    ) -> bool:
        is_alone_in_partition = cls.is_node_alone_in_partition(
            node, partition_list, filter_fn=is_not_qdq_node
        )
        if is_alone_in_partition:
            neutron_c = getattr(
                custom_delegation_options, "use_new_flow_neutron_c", False
            )
            return activation_supported_on_target(
                node, neutron_target_spec, use_new_flow_neutron_c=neutron_c
            )

        return True

    def convert(self, node: Node):
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)
        t_op.opcode_index = self.builder.op_code_index_for_op_type(BuiltinOperator.RELU)

        self.builder.append_operators([t_op])
