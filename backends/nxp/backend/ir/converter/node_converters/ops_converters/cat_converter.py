# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.nxp.backend.custom_delegation_options import (
    CustomDelegationOptions,
)
from executorch.backends.nxp.backend.edge_helper import previous_non_qdq_node
from executorch.backends.nxp.backend.ir.converter.conversion import translator
from executorch.backends.nxp.backend.ir.converter.conversion.translator import (
    apply_permutation_to,
    create_channels_first_to_channels_last_permutation,
)
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    _is_dequant_node,
    _is_quant_node,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options.concatenation_options import (
    Concatenation,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from executorch.backends.nxp.backend.node_format import NXP_NODE_FORMAT
from torch.fx import Node
from torch.fx.passes.infra.partitioner import Partition
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
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        if custom_delegation_options.force_delegate_cat:
            return True

        dim = CatConverter._get_normalized_dim(node)

        # Neutron requires the channels to be a multiple of `num_macs`. The channels could either be the second or the
        #  last dimension, depending on the formats of the node.
        if node.meta[NXP_NODE_FORMAT].is_channels_first():
            # During conversion to IR, the shape will be permuted to channels last, and the dimension on index
            #  `1` will end up being the channels (last dim in NHWC).
            channels_index = 1
            to_nhwc_perm = create_channels_first_to_channels_last_permutation(
                len(node.meta["val"].shape), True
            )
            dim = to_nhwc_perm.index(
                dim
            )  # Make sure the dim points to the NHWC dimension.
        else:
            # The shape will not be permuted during conversion, so the channels will remain the last dimension.
            channels_index = -1

        input_channels = [
            _get_shape(input_)[channels_index] for input_ in node.all_input_nodes
        ]
        output_channels = _get_shape(node)[channels_index]

        num_macs = neutron_target_spec.get_num_macs()
        input_shapes = [_get_shape(input_) for input_ in node.all_input_nodes]
        if any((input_channel % num_macs) != 0 for input_channel in input_channels):
            # neutron-library/src/utils/NeutronLibraryInterrogation.cpp#1492

            # If all input shapes are equal, the neutron is able to pad the last dimension of the inputs.
            if not (
                input_shapes.count(input_shapes[0]) == len(input_shapes)
                and dim == len(input_shapes[0]) - 1
            ):
                return False

        if (output_channels % num_macs) != 0:
            # neutron-library/src/utils/NeutronLibraryInterrogation.cpp#1493

            # If all input shapes are equal, the neutron is able to pad the last dimension of the output.
            if not (
                input_shapes.count(input_shapes[0]) == len(input_shapes)
                and dim == len(input_shapes[0]) - 1
            ):
                return False

        if len(node.all_input_nodes) < 2:  # Not supported on Neutron
            # TODO Try to skip the operator if this case is realistic.
            return False

        return True

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

    @classmethod
    def supports_partitioning_result(
        cls,
        node: Node,
        partition_list: list[Partition],
        custom_delegation_options: CustomDelegationOptions,
    ):
        # There is a bug in the NeutronConverter, where if none of the input dimensions before the one referenced by
        #  `dim` are `!= 1`, the `Concat` is not delegated.
        # This only happens when the inputs to the `Concat` are model inputs, and not outputs of other
        #  operators.
        cat_partition = [p for p in partition_list if node in p.nodes][0]
        cat_inputs = map(previous_non_qdq_node, node.args[0])

        if not all(
            input_.op == "call_function" and input_ in cat_partition.nodes
            for input_ in cat_inputs
        ):
            # Some inputs of the `cat` are NOT in the same partition as `cat`.
            dim = CatConverter._get_normalized_dim(node)
            input_shapes = [list(n.meta["val"].shape) for n in node.args[0]]
            if node.meta[NXP_NODE_FORMAT].is_channels_first():
                # Transform the shapes to channels last.
                to_nhwc_perm = create_channels_first_to_channels_last_permutation(
                    len(node.meta["val"].shape), True
                )
                input_shapes = [
                    apply_permutation_to(shape, to_nhwc_perm) for shape in input_shapes
                ]

                # Transform the `dim` to refer to a channels last dimension.
                dim = to_nhwc_perm.index(dim)

            for input_shape in input_shapes:
                if not any(d != 1 for d in input_shape[:dim]):
                    # Do not delegate if there are no "non-1" dimensions in the shape before the `dim` dimension.
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
