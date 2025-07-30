# Copyright (c) 2025 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from executorch.backends.nxp.backend.ir.converter.node_converter import (
    NodeConverter,
    Target,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    abs_options,
)
from torch.fx import Node
from torch.nn import Parameter


class AbsConverter(NodeConverter):
    supported_targets = [Target.RT700]

    @staticmethod
    def _is_supported_in_IR(
        node: Node, parameters_mapping: dict[str, Parameter]
    ) -> bool:
        return True

    def convert(self, node: Node):
        """Convert 'aten::abs' operator to TFLite 'Abs'."""
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)

        t_op.builtin_options = abs_options.Abs()
        self.builder.append_operators([t_op])
