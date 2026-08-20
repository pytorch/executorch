# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NeutronTargetSpec,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options.exp_options import (
    Exp,
)
from torch.fx import Node
from torch.nn import Parameter


class ExpConverter(NodeConverter):

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
        # Requirements specified by the new Neutron flow documentation.
        # Input and Output must be INT8/UINT8.
        if not NodeConverter.uses_quantization_type_for_io(
            node,
            supported_types=[torch.int8, torch.uint8],
            input_indices=[0],
            output_indices=[0],
        ):
            return False
        return True

    def convert(self, node: Node):
        """Convert the `aten.exp.default` operator to Neutron IR `Exp`.
        The schema is:
            aten::exp(
                Tensor self
            ) -> Tensor
        """

        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)
        t_op.builtin_options = Exp()

        self.builder.append_operators([t_op])
