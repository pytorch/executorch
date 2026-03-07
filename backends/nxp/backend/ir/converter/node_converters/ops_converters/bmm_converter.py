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
    def _get_channels_last_shape(node: Node) -> list[int]:
        input_shape = node.meta["val"].shape

        if node.meta[NXP_NODE_FORMAT].is_channels_first():
            input_shape = translator.apply_permutation_to(
                input_shape,
                translator.create_channels_first_to_channels_last_permutation(
                    len(input_shape)
                ),
            )

        return input_shape

    @staticmethod
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        _, w1, c1 = BMMConverter._get_channels_last_shape(node.args[0])
        _, w2, c2 = BMMConverter._get_channels_last_shape(node.args[1])
        is_ch_first_1 = node.args[0].meta[NXP_NODE_FORMAT].is_channels_first()
        is_ch_first_2 = node.args[1].meta[NXP_NODE_FORMAT].is_channels_first()

        # This combination of node formats is not supported on Neutron (`adj_x = True`, `adj_y = False`),
        # but it should never happen because both input tensors are expected to share the same format.
        if is_ch_first_1 and not is_ch_first_2:
            return False

        num_macs = neutron_target_spec.get_num_macs()

        # The Neutron converter requires that every dimension participating in a
        # multiplication is divisible by NUM_MACS. If any of the relevant dimensions
        # (w1, c1, w2, c2) violates this constraint, the pattern is not supported.
        if not all(m % num_macs == 0 for m in [w1, c1, w2, c2]):
            return False

        return True

    def convert(self, node: Node):
        """Convert the `aten.bmm` operator to TFLite `BatchMatMul`."""
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)

        # We set `adj_x = adj_y = True` when the inputs are in channels‑first format so
        # that TFLite internally transposes them to channels‑last. In that case, the
        # output also becomes channels‑last, so we need to transpose it back to
        # channels‑first afterward.
        #
        # We set `asymmetric_quantize_inputs = False`. Neutron ignores this parameter
        # entirely, so its value does not affect delegation and can be set arbitrarily.
        is_ch_first_1 = node.args[0].meta[NXP_NODE_FORMAT].is_channels_first()
        is_ch_first_2 = node.args[1].meta[NXP_NODE_FORMAT].is_channels_first()
        t_op.builtin_options = batch_mat_mul_options.BatchMatMul(is_ch_first_1, is_ch_first_2, False)

        x1 = t_op.tmp_inputs[0]
        x2 = t_op.tmp_inputs[1]
        y = t_op.tmp_outputs[0]

        # Assign the operator its TFLite inputs and outputs
        t_op.tmp_inputs = [x1, x2]
        t_op.tmp_outputs = [y]

        ops = OpsList(middle_op=t_op)

        # Transpose back to channels-last if needed.
        if node.meta[NXP_NODE_FORMAT].is_channels_first():
            tensor_rank = input_rank(node, 0)
            perm = translator.create_channels_first_to_channels_last_permutation(
                tensor_rank, return_list=True
            )
            ops.add_post(self.builder.create_transpose_operator_after(t_op, 0, perm))

        self.builder.append_operators(ops.flatten())
