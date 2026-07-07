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
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    add_options,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from torch.fx import Node
from torch.nn import Parameter


class AddTensorConverter(NodeConverter):
    @staticmethod
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        if not NodeConverter.at_least_one_input_shape_matches_the_output_shape(node):
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

        if hasattr(node.kwargs, "alpha"):
            return False

        return True

    def convert(self, node: Node):
        """Convert 'add_tensor' operator to NeutronIR 'Add'.
        The ExecuTorch schema is:
            add.Tensor(Tensor self, Tensor other, Scalar alpha=1)
        """
        self.assert_convertible(node)
        t_op = self._create_tflite_op_with_io_tensors(node)
        t_op.builtin_options = add_options.Add()

        ops = OpsList(middle_op=t_op)
        # Create additional ops in case of shape broadcasting
        ops.add_pre(self.builder.ensure_correct_broadcasting(t_op, t_op.tmp_outputs[0]))
        self.builder.append_operators(ops.flatten())
