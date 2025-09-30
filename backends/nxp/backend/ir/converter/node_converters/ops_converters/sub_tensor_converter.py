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
    Target,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    sub_options,
)
from torch.fx import Node
from torch.nn import Parameter


class SubTensorConverter(NodeConverter):
    @staticmethod
    def _is_supported_on_target(
        node: Node,
        target: Target,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        match target:
            case Target.RT700:
                if node_uses_shape_broadcasting(node):
                    # Shape broadcasting may require the addition of `Transpose` ops during conversion.
                    return False

                return True

            case _:
                return False

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

    # sub.Tensor Node format: (Tensor self, Tensor other, *, Scalar alpha=1)
    def convert(self, node: Node):
        """Convert 'sub_tensor' operator to NeutronIR 'Sub'."""
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)

        t_op.builtin_options = sub_options.Sub()
        self.builder.append_operators([t_op])
