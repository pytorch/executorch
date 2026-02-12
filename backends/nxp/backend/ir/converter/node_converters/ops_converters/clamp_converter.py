# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.backend.edge_helper import try_get_arg
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    is_not_qdq_node,
    NodeConverter,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from tflite import BuiltinOperator
from torch.fx import Node
from torch.fx.passes.infra.partitioner import Partition
from torch.nn import Parameter


class ClampConverter(NodeConverter):
    SUPPORTED_CLAMP_BOUNDS = [
        (-1, 1),  # ReluN1To1
        (0, 1),  # Relu0To1
        (0, 6),  # Relu6
        (0, None),  # Relu
    ]

    # noinspection PyShadowingBuiltins
    @staticmethod
    def _get_clamp_bounds(clamp_node: Node) -> tuple[float | None, float | None]:
        """Extract min and max bounds from `aten.clamp.default` node."""
        min = try_get_arg(clamp_node, 1)
        max = try_get_arg(clamp_node, 2)
        return min, max

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        # No NeutronIR-specific restrictions.
        return True

    @staticmethod
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        bounds = ClampConverter._get_clamp_bounds(node)

        # Only some specific bounds are supported on the target hardware.
        if bounds not in ClampConverter.SUPPORTED_CLAMP_BOUNDS:
            return False

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
        bounds = cls._get_clamp_bounds(node)

        if bounds in [(0, 6), (0, None)]:  # Relu6  # Relu
            # If this is the only operator in the partition, NeutronConverter will not create a NeutronNode for some
            #  reason.
            clamp_partitions = [p for p in partition_list if node in p.nodes]
            if len(clamp_partitions) != 1:
                return False  # Should never happen

            clamp_partition = clamp_partitions[0]
            non_q_dq_partition_nodes = list(
                filter(is_not_qdq_node, clamp_partition.nodes)
            )
            if len(non_q_dq_partition_nodes) <= 1:
                return False  # This would be the only node in the partition, which would cause a crash later on.

        return True

    def convert(self, node: Node):
        """Convert the `aten.clamp.default` operator to Neutron IR `Relu*` operators.
        The schema is:
            aten::clamp(
                Tensor self,
                Scalar? min=None,
                Scalar? max=None
            ) -> Tensor
        """
        self.assert_convertible(node)

        bounds = self._get_clamp_bounds(node)

        match bounds:
            case [-1, 1]:
                # ReluN1To1
                builtin_op = BuiltinOperator.RELU_N1_TO_1

            case [0, 1]:
                # Relu0To1
                builtin_op = BuiltinOperator.RELU_0_TO_1

            case [0, 6]:
                # Relu6
                builtin_op = BuiltinOperator.RELU6

            case [0, None]:
                # Relu
                builtin_op = BuiltinOperator.RELU

            case _:
                # Should never happen
                raise ValueError(
                    f"NXP backend: Clamp operator {node} was incorrectly selected for delegation."
                )

        t_op = self._create_tflite_op_with_io_tensors(node)

        # noinspection PyTypeChecker,PyUnboundLocalVariable
        t_op.opcode_index = self.builder.op_code_index_for_op_type(builtin_op)
        self.builder.append_operators([t_op])
