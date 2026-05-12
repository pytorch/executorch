# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options.leaky_relu_options import (
    LeakyRelu,
)

from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from torch.fx import Node
from torch.nn import Parameter


class LeakyReluConverter(NodeConverter):

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        return True

    @staticmethod
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        if neutron_target_spec.use_new_flow_neutron_c:
            # Requirements specified by the new Neutron flow documentation.

            if not NodeConverter.uses_quantization_type_for_io(
                node,
                supported_types=[torch.int8, torch.uint8],
                input_indices=[0],
                output_indices=[0],
            ):
                return False

            return True
        else:

            return True

    def convert(self, node: Node):
        """Convert the `aten.leaky_relu.default` operator to Neutron IR `LeakyRelu`.
        The schema is:
            aten::leaky_relu(
                Tensor self,
                Scalar negative_slope=0.01
            ) -> Tensor
        """
        self.assert_convertible(node)

        alpha = node.args[1] if len(node.args) > 1 else 0.01

        t_op = self._create_tflite_op_with_io_tensors(node)
        t_op.builtin_options = LeakyRelu(alpha)

        self.builder.append_operators([t_op])
