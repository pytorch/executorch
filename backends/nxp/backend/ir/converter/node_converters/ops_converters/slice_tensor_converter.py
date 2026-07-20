# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from executorch.backends.nxp.backend.edge_helper import input_tensor
from executorch.backends.nxp.backend.ir.converter.conversion import translator
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    slice_options,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from torch.fx import Node
from torch.nn import Parameter


class SliceTensorConverter(NodeConverter):
    @staticmethod
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        supported_types = [torch.int8, torch.uint8]
        if not NodeConverter.uses_quantization_type_for_io(
            node, supported_types, [0], [0]
        ):
            return False

        return True

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        args = node.args
        if len(args) != 4:
            return False

        dim, start, end = SliceTensorConverter._get_clipped_slice_args(node)
        input_rank = len(input_tensor(node, 0).shape)

        # Check "dim" out of bounds
        if dim >= input_rank or abs(dim) > input_rank:
            return False

        # Check invalid combination of "start" and "end" parameters
        if start >= end:
            return False

        return True

    def _convert_to_slice(self, t_op, main_input, input_rank, dim, start, end) -> None:
        # Prepare the TFLite parameters 'begin' and 'size' tensors
        begin = [0] * input_rank  # By default, start the slice at 0
        size = (
            main_input.shape.vector.copy()
        )  # By default, end the slice at the end of the dimension

        size[dim] = max(end - start, 0)
        begin[dim] = start

        begin_tensor = self.builder.create_tensor_for_data(
            np.asarray(begin, np.int32), "begin"
        )
        size_tensor = self.builder.create_tensor_for_data(
            np.asarray(size, np.int32), "size"
        )

        t_op.tmp_inputs = [main_input, begin_tensor, size_tensor]
        t_op.builtin_options = slice_options.Slice()

        self.builder.append_operators([t_op])

    Dim = Start = End = int

    @staticmethod
    def _get_clipped_slice_args(node: Node) -> tuple[Dim, Start, End]:
        input_shape = input_tensor(node, 0).shape
        _, dim, start, end = node.args
        sliced_tensor_rank = input_shape[dim]

        # convert numbering `from the end` to `from the beginning`, ie. normalize
        end = end + sliced_tensor_rank if end < 0 else end
        start = start + sliced_tensor_rank if start < 0 else start

        end = int(np.clip(end, 0, sliced_tensor_rank))
        start = int(np.clip(start, 0, sliced_tensor_rank))

        return dim, start, end

    def convert(self, node: Node):
        """Convert 'slice_tensor' operator to NeutronIR 'Slice'."""
        self.assert_convertible(node)
        t_op = self._create_tflite_op_with_io_tensors(node)
        inputs = t_op.tmp_inputs[0]
        rank = inputs.rank

        dim, start, end = self._get_clipped_slice_args(node)

        if t_op.tmp_inputs[0].tensor_format.is_channels_last():
            dim = translator.create_channels_last_to_channels_first_permutation(
                t_op.tmp_inputs[0].rank
            )[dim]

        self._convert_to_slice(t_op, inputs, rank, dim, start, end)
