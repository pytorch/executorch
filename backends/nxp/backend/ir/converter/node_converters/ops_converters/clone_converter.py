# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.nxp.backend.edge_helper import (
    get_non_qdq_parent,
    get_non_qdq_users,
)
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    is_not_qdq_node,
    NodeConverter,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from torch.fx import Node
from torch.fx.passes.infra.partitioner import Partition
from torch.nn import Parameter


def _has_supported_memory_format(node: Node) -> bool:
    """The node can either represent an `aten.clone` or a `dim_order_ops._clone_dim_order` operator."""
    memory_format = node.kwargs.get("memory_format", None)  # Attribute of `aten.clone`.
    dim_order = node.kwargs.get(
        "dim_order", None
    )  # Attribute of `dim_order_ops._clone_dim_order`.

    if (memory_format, dim_order) == (torch.preserve_format, None):
        # The operator does nothing (e.g. originated as a `Dropout`).
        return True

    contiguous_dim_order = list(range(len(node.meta["val"].shape)))
    if (memory_format, dim_order) in [
        (torch.contiguous_format, None),
        (None, contiguous_dim_order),
    ]:
        # Sometimes there is a `permute_copy` (Transpose) in Executorch, which doesn't actually permute the data in
        #  memory. Instead, it just changes the `strides` (memory format) to match the permutation. Then, some
        #  following operator may or may not support the particular strides (e.g. `mul` supports anything but
        #  `view_copy` does not), so the `clone` may be inserted to actually permute the data in memory to the
        #  `contiguous` format. This is purely an Executorch issue, and there is no equivalent system in NeutronIR.
        #  In NeutronIR, every tensor is stored in memory exactly as its shape suggests. Therefore, the `clone` can
        #  simply be omitted.
        return True

    return False


class CloneConverter(NodeConverter):
    """
    This converter is responsible for converting both edge operators:
    - aten.clone.default
    - dim_order_ops._clone_dim_order.default
    """

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        return _has_supported_memory_format(node)

    @classmethod
    def supports_partitioning_result(
        cls,
        node: Node,
        partition_list: list[Partition],
        custom_delegation_options: CustomDelegationOptions,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
    ) -> bool:
        clone_partitions = [
            partition for partition in partition_list if node in partition.nodes
        ]
        assert len(clone_partitions) == 1
        non_q_dq_partition_nodes = list(
            filter(
                is_not_qdq_node, (clone_partition_nodes := clone_partitions[0].nodes)
            )
        )

        if len(non_q_dq_partition_nodes) == 1:
            # The `clone` cannot be the only node in a partition, as it will get converted into a no-op.
            return False

        # If the `clone` will consume and input or produce an output of a delegated partition, it's input/output dim
        #  order must be either `contiguous`, or `channels last` as those are the only 2 options supported by NXP
        #  runtime.
        rank = len(node.meta["val"].shape)
        contiguous_dim_order = list(range(rank))
        channels_last_dim_order = [0] + list(range(2, rank)) + [1]
        parent_node = get_non_qdq_parent(node)
        user_nodes = get_non_qdq_users(node)
        if parent_node not in clone_partition_nodes:
            # The `clone` consumes a partition input.
            input_dim_order = list(node.args[0].meta["val"].dim_order())
            if input_dim_order not in [contiguous_dim_order, channels_last_dim_order]:
                return False

        if any(user not in clone_partition_nodes for user in user_nodes):
            # The `clone` produces a partition output.
            output_dim_order = list(node.meta["val"].dim_order())
            if output_dim_order not in [contiguous_dim_order, channels_last_dim_order]:
                return False

        return True

    def convert(self, node: Node):
        """Skip `aten.clone` operator if it has no `memory_format` specified."""
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)

        self.builder.turn_operator_to_identity(t_op)
        self.builder.append_operators([t_op])
