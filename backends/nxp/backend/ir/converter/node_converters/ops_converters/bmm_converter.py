# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch

from executorch.backends.nxp.backend.data_format import NXP_NODE_FORMAT
from executorch.backends.nxp.backend.edge_helper import (
    get_quantization_parameters_for,
    input_rank,
)
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
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        if not NodeConverter.uses_quantization_type_for_io(
            node,
            supported_types=[torch.int8, torch.uint8],
            input_indices=[0, 1],
            output_indices=[0],
        ):
            return False

        _, input_1_zp = get_quantization_parameters_for(node.args[0])
        _, input_2_zp = get_quantization_parameters_for(node.args[1])
        if not (input_1_zp == input_2_zp == 0):
            # Neutron requirement.
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
