# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.backend.edge_helper import input_rank
from executorch.backends.nxp.backend.ir.converter.conversion.common import OpsList
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    NodeConverter,
    Target,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    fully_connected_options,
)
from torch.fx import Node
from torch.nn import Parameter


class AddMMConverter(NodeConverter):
    """Convert the `aten.addmm` operator to TFLite `FullyConnected` with a bias input."""

    @staticmethod
    def _is_supported_in_IR(
        node: Node, parameters_mapping: dict[str, Parameter]
    ) -> bool:
        if len(node.all_input_nodes) != 3:
            return False

        # The weights must be 2D.
        if input_rank(node, 2) != 2:
            return False

        return True

    supported_targets = [Target.RT700]

    def convert(self, node: Node):
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)
        t_op.builtin_options = fully_connected_options.FullyConnected(
            keep_num_dims=True
        )

        bias = t_op.tmp_inputs[0]
        x = t_op.tmp_inputs[1]
        w = t_op.tmp_inputs[2]
        y = t_op.tmp_outputs[0]

        # Assign the operator its TFLite inputs and outputs
        t_op.tmp_inputs = [x, w, bias]
        t_op.tmp_outputs = [y]

        ops = OpsList(middle_op=t_op)

        # The `aten.addmm` uses main input with shape [M, N] and the weights have the shape [N, O].
        # TFLite `FullyConnected` requires the weights to have shape [O, N] (if the main input has shape [M, N]).
        # Insert a `Transpose` operator to permute the weights to achieve correct conversion. (The `Transpose` will not
        #  be present in the output model if the weights are static.)
        ops.add_pre(self.builder.create_transpose_operator_before(t_op, 1, [1, 0]))

        self.builder.append_operators(ops.flatten())
