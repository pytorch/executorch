# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.backend.ir.converter.conversion.common import (
    node_uses_shape_broadcasting,
)
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    mul_options,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from torch.fx import Node
from torch.nn import Parameter


class MulTensorConverter(NodeConverter):
    @staticmethod
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        if node_uses_shape_broadcasting(node):
            # Shape broadcasting may require the addition of `Transpose` ops during conversion.
            return False

        node_shape = node.meta["val"].shape

        # Check that at least one dimension is divisible by number of MACS
        # or all dimensions are equal to one
        # Otherwise Neutron cannot convert it
        dim_divisible = any(s % 8 == 0 for s in node_shape) or all(
            s == 1 for s in node_shape
        )
        return dim_divisible

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        if len(node.args) != 2:
            return False

        return True

    # mul.Tensor Node format: (Tensor self, Tensor other, *)
    def convert(self, node: Node):
        """Convert 'mul_tensor' operator to NeutronIR 'Mul'."""
        self.assert_convertible(node)
        t_op = self._create_tflite_op_with_io_tensors(node)
        t_op.builtin_options = mul_options.Mul()

        self.builder.append_operators([t_op])
