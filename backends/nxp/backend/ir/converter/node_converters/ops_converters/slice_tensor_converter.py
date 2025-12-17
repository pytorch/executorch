# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from executorch.backends.nxp.backend.edge_helper import input_tensor
from executorch.backends.nxp.backend.ir.converter.conversion import translator
from executorch.backends.nxp.backend.ir.converter.conversion.common import OpsList
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    slice_options,
)
from executorch.backends.nxp.backend.neutron_operator_support import (
    transposition_is_supported_on_neutron,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from executorch.backends.nxp.backend.node_format import NXP_NODE_FORMAT
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
        # Provisional solution - slice conversion works for neutron software 2.2.1+
        neutron_flavor = neutron_target_spec.neutron_target.__module__.split(".")[0]
        if neutron_flavor != "neutron_converter_SDK_25_12":
            return False

        input_shape = input_tensor(node, 0).shape
        dim = node.args[1]
        if node.args[0].meta[NXP_NODE_FORMAT].is_channels_first():
            dim = translator.create_channels_last_to_channels_first_permutation(
                len(input_shape)
            )[dim]
            input_shape = translator.apply_permutation_to(
                input_shape,
                translator.create_channels_first_to_channels_last_permutation(
                    len(input_shape)
                ),
            )
        input_rank = len(input_shape)

        # Slicing is only allowed along the channel dimension.
        # Therefore, we must verify that Neutron supports swapping the channel dimension
        # with the dimension intended for slicing.
        if dim != -1 and dim != input_rank - 1:
            perm = list(range(0, input_rank))
            perm[dim], perm[-1] = perm[-1], perm[dim]

            if not transposition_is_supported_on_neutron(
                list(input_shape), perm, neutron_target_spec
            ):
                return False

        # The shape of dimension that we want to slice must be divisible by num_macs
        num_macs = neutron_target_spec.get_num_macs()
        return input_shape[dim] % num_macs == 0

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

        # We can slice only the channels dimension
        # So we swap the sliced dimension with the channels dimension
        begin[-1], begin[dim] = begin[dim], begin[-1]
        size[-1], size[dim] = size[dim], size[-1]

        begin_tensor = self.builder.create_tensor_for_data(
            np.asarray(begin, np.int32), "begin"
        )
        size_tensor = self.builder.create_tensor_for_data(
            np.asarray(size, np.int32), "size"
        )

        t_op.tmp_inputs = [main_input, begin_tensor, size_tensor]
        t_op.builtin_options = slice_options.Slice()
        ops = OpsList(middle_op=t_op)

        # If slicing along non-channels dimension, we need to swap it with channels dimension.
        # Otherwise Neutron will not convert it.
        if dim != -1 and dim != input_rank - 1:
            # Create permutation for swapping
            perm = list(range(0, input_rank))
            perm[dim], perm[-1] = perm[-1], perm[dim]

            # Insert forward and backward transpose
            ops.add_pre(self.builder.create_transpose_operator_before(t_op, 0, perm))
            ops.add_post(self.builder.create_transpose_operator_after(t_op, 0, perm))

        self.builder.append_operators(ops.flatten())

    Dim = Start = End = int

    @staticmethod
    def _get_clipped_slice_args(node: Node) -> tuple[Dim, Start, End]:
        input_shape = input_tensor(node, 0).shape
        _, dim, start, end = node.args
        sliced_tensor_rank = input_shape[dim]

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
