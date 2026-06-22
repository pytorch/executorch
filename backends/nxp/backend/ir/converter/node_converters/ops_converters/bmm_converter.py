# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.backend.data_format import NXP_NODE_FORMAT
from executorch.backends.nxp.backend.edge_helper import input_rank
from executorch.backends.nxp.backend.ir.converter.conversion import translator
from executorch.backends.nxp.backend.ir.converter.conversion.common import OpsList
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    batch_mat_mul_options,
)
from executorch.backends.nxp.backend.neutron_operator_support import (
    transposition_is_supported_on_neutron,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from torch.fx import Node
from torch.nn import Parameter


class BMMConverter(NodeConverter):
    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        if len(node.all_input_nodes) != 2:
            return False

        if input_rank(node, 0) != 3 or input_rank(node, 1) != 3:
            return False

        return True

    @staticmethod
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        is_ch_first_1 = node.args[0].meta[NXP_NODE_FORMAT].is_channels_first()
        is_ch_first_2 = node.args[1].meta[NXP_NODE_FORMAT].is_channels_first()
        # This combination of node formats is not supported on Neutron (`adj_x = True`, `adj_y = False`),
        # but it should never happen because both input tensors are expected to share the same format.
        if is_ch_first_1 and not is_ch_first_2:
            return False

        # In case we need to insert transpose after `BatchMatMul`, we also need to check if
        # such transposition is supported.
        if node.meta[NXP_NODE_FORMAT].is_channels_first():
            tensor_shape = node.meta["val"].shape
            tensor_rank = len(tensor_shape)
            perm = translator.create_channels_first_to_channels_last_permutation(
                tensor_rank, return_list=True
            )

            tensor_shape_channels_last = [tensor_shape[i] for i in perm]
            if not transposition_is_supported_on_neutron(
                tensor_shape_channels_last, perm, neutron_target_spec
            ):
                return False

        _, d1, d2 = node.args[0].meta["val"].shape
        _, d3, d4 = node.args[1].meta["val"].shape

        # The Neutron converter requires that every dimension participating in the
        # multiplication is divisible by NUM_MACS.
        num_macs = neutron_target_spec.get_num_macs()
        if not all(m % num_macs == 0 for m in [d1, d2, d3, d4]):
            return False

        return True

    def convert(self, node: Node):
        """Convert the `aten.bmm` operator to TFLite `BatchMatMul`."""
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)

        # We set `adj_x = adj_y = True` when the inputs are in channels‑last format so
        # that TFLite internally transposes them to channels‑first. In that case, the
        # output also becomes channels‑first, so we need to transpose it back to
        # channels‑last afterward.
        #
        # We set `asymmetric_quantize_inputs = False`. Neutron ignores this parameter
        # entirely, so its value does not affect delegation and can be set arbitrarily.
        is_ch_first_1 = node.args[0].meta[NXP_NODE_FORMAT].is_channels_first()
        is_ch_first_2 = node.args[1].meta[NXP_NODE_FORMAT].is_channels_first()
        t_op.builtin_options = batch_mat_mul_options.BatchMatMul(
            is_ch_first_1, is_ch_first_2, False
        )

        x1 = t_op.tmp_inputs[0]
        x2 = t_op.tmp_inputs[1]
        y = t_op.tmp_outputs[0]

        # Assign the operator its TFLite inputs and outputs
        t_op.tmp_inputs = [x1, x2]
        t_op.tmp_outputs = [y]

        ops = OpsList(middle_op=t_op)

        # Transpose back to channels-last if needed.
        if node.meta[NXP_NODE_FORMAT].is_channels_first():
            tensor_rank = len(node.meta["val"].shape)
            perm = translator.create_channels_first_to_channels_last_permutation(
                tensor_rank, return_list=True
            )
            ops.add_post(self.builder.create_transpose_operator_after(t_op, 0, perm))

        self.builder.append_operators(ops.flatten())
