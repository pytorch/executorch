# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.nxp.backend.custom_delegation_options import (
    CustomDelegationOptions,
)
from executorch.backends.nxp.backend.ir.converter.conversion import translator
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    _is_dequant_node,
    _is_quant_node,
    NodeConverter,
    Target,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options.concatenation_options import (
    Concatenation,
)
from torch.fx import Node
from torch.nn import Parameter


def _get_shape(node: torch.fx.Node) -> list[int]:
    return node.meta["val"].shape


class CatConverter(NodeConverter):

    @staticmethod
    def _get_normalized_dim(node: torch.fx.Node) -> int:
        dim = node.args[1] if len(node.args) >= 2 else 0  # Default `dim` value.
        rank = len(_get_shape(node))
        if dim < 0:
            dim += rank

        if not (0 <= dim < rank):
            raise RuntimeError("`Cat` operator has invalid `dim`.")

        return dim

    @staticmethod
    def _all_io_shares_quantization_parameters(node: Node) -> bool:
        post_node = list(node.users.keys())[0]
        if not _is_quant_node(post_node):
            return False
        output_zp, output_scale, output_type = (
            post_node.args[1],
            post_node.args[2],
            post_node.args[5],
        )

        for input_node in node.args[0]:
            if not _is_dequant_node(input_node):
                return False

            input_zp, input_scale, input_type = (
                input_node.args[1],
                input_node.args[2],
                input_node.args[5],
            )
            if (input_zp, input_scale, input_type) != (
                output_zp,
                output_scale,
                output_type,
            ):
                return False

        return True

    @staticmethod
    def _is_supported_on_target(
        node: Node,
        target: Target,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        if custom_delegation_options.force_delegate_cat:
            return True

        match target:
            case Target.RT700:
                dim = CatConverter._get_normalized_dim(node)

                # neutron-library/src/utils/NeutronLibraryInterrogation.cpp#1491
                if dim == 0:
                    return False

                # Neutron requires the channels to be a multiple of `8`. The channels could either be the second or the
                #  last dimension, depending on the formats of the node. The format, however, cannot be determined
                #  during conversion, as it depends on what other nodes are delegated.
                input_channels = [
                    # The second dimension is the channels in PyTorch. If the inputs/output are not channels first, it
                    #  will still be the channels in the IR.
                    _get_shape(input_)[1]
                    for input_ in node.all_input_nodes
                ] + [
                    # If the inputs/outputs are channels first, the last dimension will be the channels.
                    _get_shape(input_)[-1]
                    for input_ in node.all_input_nodes
                ]
                if any((input_channel % 8) != 0 for input_channel in input_channels):
                    # neutron-library/src/utils/NeutronLibraryInterrogation.cpp#1492
                    return False

                output_channels = [_get_shape(node)[1], _get_shape(node)[-1]]
                # neutron-library/src/utils/NeutronLibraryInterrogation.cpp#1493
                if any((out_c % 8) != 0 for out_c in output_channels):
                    return False

                if len(node.all_input_nodes) < 2:  # Not supported on Neutron
                    # TODO Try to skip the operator if this case is realistic.
                    return False

                return True

            case _:
                return False

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        if not CatConverter._all_io_shares_quantization_parameters(node):
            # The IR requires all inputs to have the same quantization parameters as the output.
            # The quantizer should quantize the operator so that this case does not happen.
            return False

        return True

    def convert(self, node: Node):
        """Convert the 'aten.cat' operator to TFLite 'Concatenation'."""
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)

        dim = self._get_normalized_dim(node)  # Also checks the validity of `dim`.

        if t_op.tmp_inputs[0].tensor_format.is_channels_last():
            dim = translator.create_channels_last_to_channels_first_permutation(
                t_op.tmp_inputs[0].rank
            )[dim]

        t_op.builtin_options = Concatenation(dim)
        self.builder.append_operators([t_op])
