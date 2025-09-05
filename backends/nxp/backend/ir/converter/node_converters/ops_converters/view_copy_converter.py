# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from executorch.backends.nxp.backend.edge_helper import (
    input_tensor,
    output_tensor,
    tensor_rank,
)
from executorch.backends.nxp.backend.ir.converter import quantization_utils
from executorch.backends.nxp.backend.ir.converter.conversion.common import OpsList
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.shared.reshape_transposition import (
    ensure_reshape_transposition,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    reshape_options,
)
from torch.fx import Node
from torch.nn import Parameter


class ViewCopyConverter(NodeConverter):

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        x = input_tensor(node, 0)
        y = output_tensor(node)

        flat_input_size = ViewCopyConverter._safe_compute_flat_size(list(x.size()))
        flat_output_size = ViewCopyConverter._safe_compute_flat_size(list(y.size()))

        if tensor_rank(y) >= 8 or flat_input_size != flat_output_size:
            return False

        return True

    @staticmethod
    def _safe_compute_flat_size(shape: list[int | str]) -> int:
        """Compute the flat size of a tensor with given shape. Strings and negative dimensions are treated as '1'.

        :param shape: Shape of the tensor. Can include integers and strings.
        :return: The flat size of the tensor.
        """
        flat_size = 1
        for dim in shape:
            if isinstance(dim, int) and dim > 1:
                flat_size *= dim

        return flat_size

    def convert(self, node: Node):
        """Convert the `aten.view_copy` operator to TFLite `Reshape`."""
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        ops = OpsList(middle_op=t_op)

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
                "ViewCopyConverter: Q-params of input and output doesn't match. This "
                "indicates error in quantizer."
            )

        new_shape = ensure_reshape_transposition(self.builder, ops)

        # Create the TFLite Reshape with the new shape
        t_op.builtin_options = reshape_options.Reshape(new_shape)

        # Required by neutron-converter, but it will remove this tensor in optimization phase
        new_shape_tensor = self.builder.create_tensor_for_data(
            np.asarray(new_shape, dtype=np.int32), "new_shape"
        )
        t_op.tmp_inputs.append(new_shape_tensor)

        self.builder.append_operators(ops.flatten())
