# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch

from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NeutronTargetSpec,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    abs_options,
)
from torch.fx import Node
from torch.nn import Parameter


class AbsConverter(NodeConverter):

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

        if custom_delegation_options.use_new_flow_neutron_c:
            # Requirements specified by the new Neutron flow documentation.

            supported_types = [torch.int8, torch.uint8]
            if not NodeConverter.uses_quantization_type_for_io(
                node, supported_types, [0], [0]
            ):
                return False

        return True

    def convert(self, node: Node):
        """Convert 'aten::abs' operator to TFLite 'Abs'."""
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)

        t_op.builtin_options = abs_options.Abs()
        self.builder.append_operators([t_op])
