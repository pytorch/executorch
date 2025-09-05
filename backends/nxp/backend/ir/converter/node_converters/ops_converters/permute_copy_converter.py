# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from executorch.backends.nxp.backend.ir.converter import quantization_utils
from executorch.backends.nxp.backend.ir.converter.conversion.common import OpsList
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    transpose_options,
)
from torch.fx import Node
from torch.nn import Parameter


class PermuteCopyConverter(NodeConverter):

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        return True

    def convert(self, node: Node):
        """Convert the `aten.permute_copy` operator to TFLite `Transpose`."""
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)
        t_op.builtin_options = transpose_options.Transpose()

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        if (
            x.quantization is not None
            and y.quantization is None
            and "cluster" in node.meta
        ):
            # We know this node is part of QDQ cluster, so we can propagate quantization to inputs of "call_function"
            # node of this cluster.
            quantization_utils.propagate_quantization(x, y)

            y.type = x.type
            assert x.quantization == y.quantization, (
                "PermuteCopyConverter: Q-params of input and output doesn't "
                "match. This indicates error in quantizer."
            )

        perm = np.array(node.args[1], "int32")
        perm_tensor = self.builder.create_tensor_for_data(perm, "perm")

        # Assign the operator its TFLite inputs and outputs
        t_op.tmp_inputs = [x, perm_tensor]
        t_op.tmp_outputs = [y]

        ops_to_add = OpsList(middle_op=t_op)

        self.builder.append_operators(ops_to_add.flatten())
