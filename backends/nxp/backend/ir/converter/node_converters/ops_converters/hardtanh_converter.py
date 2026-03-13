# Copyright 2025-2026 NXP
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
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from torch.fx import Node
from torch.nn import Parameter


class HardTanhConverter(NodeConverter):

    # Maps possible input parameters of HardTanh to equivalent ReLU-based operators supported by TFLite.
    SUPPORTED_MODES_MAP = {
        (0.0, 6.0): BuiltinOperator.RELU6,
        (-1.0, 1.0): BuiltinOperator.RELU_N1_TO_1,
        (0.0, 1.0): BuiltinOperator.RELU_0_TO_1,
        (0.0, float("inf")): BuiltinOperator.RELU,
    }

    # Maps possible modes of HardTanh to equivalent ReLU bounds.
    SUPPORTED_BOUNDS_MAP = {
        "ReluN1To1": (-1.0, 1.0),
        "Relu0To1": (0.0, 1.0),
        "Relu6": (0.0, 6.0),
        "Relu": (0.0, float("inf")),
    }

    @staticmethod
    def _get_hardtanh_bounds(node: Node) -> tuple[int, int]:
        args = node.args

        match len(args):
            case 1:
                min_val = -1
                max_val = 1

            case 2:
                min_val = args[1]
                max_val = 1

            case 3:
                min_val = args[1]
                max_val = args[2]

            case _:
                # should not occur
                min_val = 0
                max_val = 1

        return min_val, max_val

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        bounds = HardTanhConverter._get_hardtanh_bounds(node)
        return bounds in HardTanhConverter.SUPPORTED_MODES_MAP.keys()

    @classmethod
    def supports_partitioning_result(
        cls,
        node: Node,
        partition_list: list[Partition],
        custom_delegation_options: CustomDelegationOptions,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
    ) -> bool:
        bounds = HardTanhConverter._get_hardtanh_bounds(node)

        if bounds in [
            cls.SUPPORTED_BOUNDS_MAP["Relu"],
            cls.SUPPORTED_BOUNDS_MAP["Relu6"],
        ]:
            is_alone_in_partition = cls.is_node_alone_in_partition(
                node, partition_list, filter_fn=is_not_qdq_node
            )
            if is_alone_in_partition:
                return activation_supported_on_target(node, neutron_target_spec)

        return True

    def convert(self, node: Node):
        """Convert 'aten::hardtanh' to it's supported ReLU equivalent."""
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)

        bounds = HardTanhConverter._get_hardtanh_bounds(node)

        op = self.SUPPORTED_MODES_MAP[bounds]
        t_op.opcode_index = self.builder.op_code_index_for_op_type(op)

        self.builder.append_operators([t_op])
