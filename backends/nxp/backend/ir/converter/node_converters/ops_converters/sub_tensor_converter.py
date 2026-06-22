# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.nxp.backend.data_format import NXP_NODE_FORMAT
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    sub_options,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from torch.fx import Node
from torch.nn import Parameter


class SubTensorConverter(NodeConverter):
    @staticmethod
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        if not NodeConverter.at_least_one_input_shape_matches_the_output_shape(node):
            return False

        # If one input is in channel first and ranks of input tensors are not equal, we need to add Transposes
        # Transpose is currently not supported for new flow
        if any(
            input_node.meta[NXP_NODE_FORMAT].is_channels_first()
            for input_node in node.all_input_nodes
        ) and NodeConverter._node_inputs_ranks_not_equal(node):
            return False

        supported_types = [torch.int8, torch.uint8]
        if not NodeConverter.uses_quantization_type_for_io(
            node, supported_types, [0, 1], [0]
        ):
            return False

        return True

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        if len(node.args) != 2:
            return False

        # The `alpha` attribute can be represented by adding an extra `Mul` operator.
        #  However, this is not implemented as `alpha` is rarely used.
        if hasattr(node.kwargs, "alpha"):
            return False

        return True

    def convert(self, node: Node):
        """Convert 'sub_tensor' operator to NeutronIR 'Sub'.
        The ExecuTorch schema is:
            sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1)
        """

        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)

        t_op.builtin_options = sub_options.Sub()
        self.builder.append_operators([t_op])
