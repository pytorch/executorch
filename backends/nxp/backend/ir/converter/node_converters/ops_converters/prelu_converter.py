# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from executorch.backends.nxp.backend.node_format import NXP_NODE_FORMAT
from torch.fx import Node
from torch.nn import Parameter


class PReLUConverter(NodeConverter):
    @staticmethod
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        node_shape = node.meta["val"].shape
        rank = len(node_shape)

        # According to Neutron spec., PReLU can be done only on 4D tensors
        if rank != 4:
            return False

        # In this case, checking op support might be done before the node format is specified,
        # for example when checking if the op should be decomposed into simpler ops.
        # That is why the dim order cannot always be properly determined.
        possible_ch_idx, possible_h_idx, possible_w_idx = PReLUConverter._get_dim_order(
            node
        )

        # According to Neutron spec., size of channels must be divisible by num_macs.
        num_macs = neutron_target_spec.get_num_macs()
        if any(
            node_shape[channels_index] % num_macs != 0
            for channels_index in possible_ch_idx
        ):
            return False

        # According to Neutron spec., height * width cannot be greater than a given constant.
        for h_idx in possible_h_idx:
            for w_idx in possible_w_idx:
                if h_idx != w_idx and node_shape[w_idx] * node_shape[h_idx] > 4096:
                    return False

        return True

    @staticmethod
    def _get_dim_order(node: Node):
        if node.meta.get(NXP_NODE_FORMAT) is None:
            possible_ch_idx = [1, 3]
            possible_h_idx = [1, 2]
            possible_w_idx = [2, 3]

        elif node.meta[NXP_NODE_FORMAT].is_channels_first():
            possible_ch_idx = [1]
            possible_h_idx = [2]
            possible_w_idx = [3]
        else:
            possible_ch_idx = [3]
            possible_h_idx = [1]
            possible_w_idx = [2]

        return possible_ch_idx, possible_h_idx, possible_w_idx

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        if len(node.args) != 2:
            return False

        return True

    def convert(self, node: Node):
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)
        t_op.opcode_index = self.context.tflite_builder.op_code_index_for_op_type(
            BuiltinOperator.PRELU
        )

        self.builder.append_operators([t_op])
