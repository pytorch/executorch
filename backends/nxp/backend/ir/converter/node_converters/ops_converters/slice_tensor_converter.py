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
    strided_slice_options,
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
        if len(node.args) != 4:
            return False

        return True

    # def _convert_to_slice(self, t_op, main_input, input_rank, starts, ends, axes) -> None:
    #     # Prepare the TFLite parameters 'begin' and 'size'
    #     begin = [0] * input_rank  # By default, start the slice at 0
    #     size = main_input.shape.vector.copy()  # By default, end the slice at the end of the dimension

    #     for i, axis in enumerate(axes):
    #         begin[axis] = starts[i]
    #         size[axis] = ends[i] - starts[i]

    #         size[axis] = max(size[axis], 0)

    #     # Create the TFLite tensors
    #     begin_tensor = self.builder.create_tensor_for_data(np.asarray(begin, np.int32), "begin")
    #     size_tensor = self.builder.create_tensor_for_data(np.asarray(size, np.int32), "size")
    #     t_op.tmp_inputs = [main_input, begin_tensor, size_tensor]
    #     t_op.builtin_options = slice_options.Slice()

    # def _convert_to_strided_slice(
    #     self, t_op: tflite_model.Operator, main_input: tflite_model.Tensor, input_rank: int,
    #     starts: list[np.ndarray], ends: list[np.ndarray], axes: list[np.ndarray], steps: list[np.ndarray]
    # ) -> None:
    #     tf_begin = [0] * input_rank  # By default, start slice from 0
    #     tf_end = main_input.shape.vector.copy()  # By default, end slice at the end of dimension
    #     tf_strides = [1] * input_rank  # By default, step by 1

    #     for i, axis in enumerate(axes):
    #         tf_begin[axis] = starts[i]
    #         tf_end[axis] = ends[i]
    #         tf_strides[axis] = steps[i]

    #         # TFLite cannot handle situation when we're iterating down
    #         # from positive values through 0, to negative values.
    #         # noinspection PyChainedComparisons
    #         if steps[i] < 0 and starts[i] >= 0 and ends[i] < 0:
    #             # Add negative offset of dimension size and make both 'begin' and 'end' negative
    #             tf_begin[axis] = starts[i] - main_input.shape.vector[axis]
    #             tf_end[axis] = ends[i] - main_input.shape.vector[axis]

    #     begin_tensor = self.builder.create_tensor_for_data(np.asarray(tf_begin, np.int32), "begin")
    #     end_tensor = self.builder.create_tensor_for_data(np.asarray(tf_end, np.int32), "ends")
    #     strides_tensor = self.builder.create_tensor_for_data(np.asarray(tf_strides, np.int32), "strides")

    #     t_op.tmp_inputs = [main_input, begin_tensor, end_tensor, strides_tensor]
    #     t_op.builtin_options = strided_slice_options.StridedSlice()

    Dims = Starts = Ends = Steps = list[int]
    @staticmethod
    def _get_slice_arguments(slice_node: Node) -> (Dims, Starts, Ends, Steps):
        dims, starts, ends, steps = (slice_node.args)

        return (list(dims), list(starts), list(ends), list(steps)) 

    def convert(self, node: Node):
        """Convert 'slice_tensor' operator to NeutronIR 'StridedSlice' or 'Slice'."""
        self.assert_convertible(node)
        t_op = self._create_tflite_op_with_io_tensors(node)
        
        dims, starts, ends, steps = self._get_slice_arguments(node)
        # if all(step == 1 for step in steps):
        #     # self._convert_to_slice(t_op, dims, starts, ends, steps)
        # else:
        #     self._convert_to_strided_slice(t_op, dims, starts, ends, steps)

        t_op.builtin_options = strided_slice_options.StridedSlice()
        self.builder.append_operators([t_op])


        