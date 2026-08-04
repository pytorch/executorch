# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.nxp.backend.edge_helper import (
    input_rank,
    weights_are_effectively_static,
)
from executorch.backends.nxp.backend.ir.converter.conversion.common import OpsList
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    fully_connected_options,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from torch.fx import Node
from torch.nn import Parameter


class MMConverter(NodeConverter):

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        if len(node.all_input_nodes) != 2:
            return False

        # The weights must be 2D.
        if input_rank(node, 1) != 2:
            return False

        return True

    @staticmethod
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        # Main input and output must be `int8` or `uint8`.
        if not NodeConverter.uses_quantization_type_for_io(
            node, [torch.int8, torch.uint8], [0], [0]
        ):
            return False

        # Weights must be `int8`.
        if not NodeConverter.uses_quantization_type_for_io(node, [torch.int8], [1], []):
            return False

        # Weights must be static.
        if not weights_are_effectively_static(node, parameters_mapping):
            return False

        return True

    def convert(self, node: Node):
        """Convert the `aten.mm` operator to Neutron IR `FullyConnected` without a bias input.
        The schema is:
            mm(
                Tensor self,
                Tensor mat2
            ) -> Tensor
        """
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)
        t_op.builtin_options = fully_connected_options.FullyConnected()

        x = t_op.tmp_inputs[0]
        w = t_op.tmp_inputs[1]
        y = t_op.tmp_outputs[0]

        # Assign the operator its Neutron IR inputs and outputs
        t_op.tmp_inputs = [x, w]
        t_op.tmp_outputs = [y]

        ops = OpsList(middle_op=t_op)

        # The `aten.mm` uses main input with shape [M, N] and the weights have the shape [N, O].
        # Neutron IR `FullyConnected` requires the weights to have shape [O, N] (if the main input has shape [M, N]).
        # Insert a `Transpose` operator to permute the weights to achieve correct conversion. (The `Transpose` will not
        #  be present in the output model if the weights are static.)
        ops.add_pre(self.builder.create_transpose_operator_before(t_op, 1, [1, 0]))

        self.builder.append_operators(ops.flatten())
