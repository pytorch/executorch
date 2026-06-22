# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from executorch.backends.nxp.backend import edge_helper
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    sub_options,
)
from executorch.backends.nxp.backend.ir.tflite_generator.tflite_model import (
    Quantization,
    Scale,
    ZeroPoint,
)
from torch.fx import Node
from torch.nn import Parameter


class NegConverter(NodeConverter):

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        if len(node.args) != 1:
            # Should never happen
            return False

        # The conversion code below expects a per tensor quantized operator.
        scale, zp = edge_helper.get_quantization_parameters_for(node.args[0])
        match scale, zp:
            case [float(), int()]:
                pass  # Atomic quantization parameters -> per tensor quantization.
            case _:
                return False  # Everything else is unexpected.

        return True

    def convert(self, node: Node):
        """Convert 'aten.neg.default' operator to NeutronIR  0 - 'Sub'.

        The ExecuTorch schema is 'aten::neg(Tensor self) -> Tensor'
        """
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)

        x = t_op.tmp_inputs[0]

        # Extract the zero_point, to use as the first input of the `Sub`.
        scale = x.quantization.scale.vector
        zp = x.quantization.zero_point.vector
        zero_tensor = self.builder.create_tensor_for_data(np.array(zp, "int8"), "zero")
        zero_tensor.quantization = Quantization(
            scale=Scale(list(scale)), zero_point=ZeroPoint(list(zp))
        )

        # Assign the NeutronIR operator its builtin options and inputs.
        t_op.builtin_options = sub_options.Sub()
        t_op.tmp_inputs = [zero_tensor, x]

        self.builder.append_operators([t_op])
