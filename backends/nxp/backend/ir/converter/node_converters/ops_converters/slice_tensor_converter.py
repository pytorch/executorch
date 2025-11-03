# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from backends.nxp.backend.ir.converter.conversion.common import OpsList
from executorch.backends.nxp.backend.ir.converter.conversion.common import (
    node_uses_shape_broadcasting,
)
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    strided_slice_options, slice_options
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from torch.fx import Node
from torch.nn import Parameter
import numpy as np


class SliceTensorConverter(NodeConverter):
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

        return True

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        args = node.args
        if len(args) == 4:
            return True
        
        return False

    def _convert_to_slice(self, t_op, main_input, input_rank, axis, start, end) -> None:
        # Prepare the TFLite parameters 'begin' and 'size'
        begin = [0] * input_rank  # By default, start the slice at 0
        size = main_input.shape.vector.copy()  # By default, end the slice at the end of the dimension

        begin[axis] = start
        size[axis] = max(end - start, 0)

        begin[-1], begin[axis] = begin[axis], begin[-1]
        size[-1], size[axis] = size[axis], size[-1]

        begin_tensor = self.builder.create_tensor_for_data(np.asarray(begin, np.int32), "begin")
        size_tensor = self.builder.create_tensor_for_data(np.asarray(size, np.int32), "size")

        t_op.tmp_inputs = [main_input, begin_tensor, size_tensor]
        t_op.builtin_options = slice_options.Slice()
        
        ops = OpsList(middle_op=t_op)
        perm = list(range(0, input_rank))
        perm[axis], perm[-1] = perm[-1], perm[axis]
        ops.add_pre(self.builder.create_transpose_operator_before(t_op, 0, perm))
        ops.add_post(self.builder.create_transpose_operator_after(t_op, 0, perm))
        
        self.builder.append_operators(ops.flatten())

    Dim = Start = End = int
    @staticmethod
    def _get_slice_arguments(slice_node: Node) -> (Dim, Start, End):
        _, dim, start, end = slice_node.args

        return dim, start, end    

    def convert(self, node: Node):
        """Convert 'slice_tensor' operator to NeutronIR 'Slice'."""
        self.assert_convertible(node)
        t_op = self._create_tflite_op_with_io_tensors(node)
        inputs = t_op.tmp_inputs[0]
        rank = inputs.rank
        
        dim, start, end = self._get_slice_arguments(node)
        self._convert_to_slice(t_op, inputs, rank, dim, start, end)
