# Copyright (c) 2025 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.backend.ir.converter.conversion.common import (
    node_uses_shape_broadcasting,
)
from executorch.backends.nxp.backend.ir.converter.node_converter import NodeConverter
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    add_options,
)
from torch.fx import Node
from torch.nn import Parameter


class AddTensorConverter(NodeConverter):

    @staticmethod
    def _is_supported_in_IR(
        node: Node, parameters_mapping: dict[str, Parameter]
    ) -> bool:
        if len(node.args) != 2:
            return False

        if hasattr(node.kwargs, "alpha"):
            return False

        # Don't convert if broadcasting input tensors
        if node_uses_shape_broadcasting(node):
            return False

        return True

    # add.Tensor Node format: (Tensor self, Tensor other, *, Scalar alpha=1)
    def convert(self, node: Node):
        """Convert 'add_tensor' operator to TFLite 'add'."""
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)

        t_op.builtin_options = add_options.Add()
        self.builder.append_operators([t_op])
