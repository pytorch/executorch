# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.nxp.backend.ir.converter.conversion.common import OpsList
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.shared.reduce_utils import (
    convert_axes_from_attribute,
    get_dim_and_handle_io_formats,
    get_reduce_node_attrs,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    mean_options,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from torch.fx import Node
from torch.fx.passes.infra.partitioner import Partition
from torch.nn import Parameter


class MeanDimConverter(NodeConverter):

    @classmethod
    def supports_partitioning_result(
        cls,
        node: Node,
        partition_list: list[Partition],
        custom_delegation_options: CustomDelegationOptions,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
    ) -> bool:
        dim, keepdim = get_reduce_node_attrs(node)
        input_shape = node.args[0].meta["val"].shape

        is_alone_in_partition = cls.is_node_alone_in_partition(node, partition_list)

        if is_alone_in_partition and keepdim and all(input_shape[d] == 1 for d in dim):
            # The operator is a no-op, so the Neutron Converter will skip it. If it's the only node in the
            #  partition, the graph would end up empty.
            return False

        return True

    @staticmethod
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        if not NodeConverter.uses_quantization_type_for_io(
            node,
            supported_types=[torch.int8, torch.uint8],
            input_indices=[0],
            output_indices=[0],
        ):
            return False

        return True

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        if hasattr(node.kwargs, "dtype") and node.kwargs["dtype"] not in [
            torch.float32,
            torch.uint32,
            torch.uint8,
        ]:
            return False

        if not NodeConverter._has_shared_q_params_if_quantized(node):
            return False

        return True

    def convert(self, node: Node):
        """Convert the 'mean.dim' operator to NeutronIR 'Mean'.
        The ExecuTorch schema is:
            mean.dim(
                Tensor self,
                int[1]? dim,
                bool keepdim=False,
                *,
                ScalarType? dtype=None
            ) -> Tensor
        """
        self.assert_convertible(node)

        dim, keepdim = get_reduce_node_attrs(node)

        t_op = self._create_tflite_op_with_io_tensors(node)
        t_op.builtin_options = mean_options.Mean(keepdim)

        ops = OpsList(middle_op=t_op)
        dim = get_dim_and_handle_io_formats(self.builder, ops, dim, keepdim)

        convert_axes_from_attribute(t_op, self.builder, dim)
        self.builder.append_operators(ops.flatten())
