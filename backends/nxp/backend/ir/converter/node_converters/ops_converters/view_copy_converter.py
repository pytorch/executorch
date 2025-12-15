# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from executorch.backends.nxp.backend.edge_helper import (
    get_non_qdq_users,
    input_tensor,
    output_tensor,
    tensor_rank,
)
from executorch.backends.nxp.backend.ir.converter import quantization_utils
from executorch.backends.nxp.backend.ir.converter.conversion.common import OpsList
from executorch.backends.nxp.backend.ir.converter.conversion.translator import (
    apply_permutation_to,
    create_channels_first_to_channels_last_permutation,
    create_channels_last_to_channels_first_permutation,
)
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    is_not_qdq_node,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.shared.reshape_transposition import (
    ensure_reshape_transposition,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    reshape_options,
)
from executorch.backends.nxp.backend.neutron_operator_support import (
    transposition_is_supported_on_neutron,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from executorch.backends.nxp.backend.node_format import NXP_NODE_FORMAT
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Node
from torch.fx.passes.infra.partitioner import Partition
from torch.nn import Parameter


class ViewCopyConverter(NodeConverter):

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        x = input_tensor(node, 0)
        y = output_tensor(node)

        flat_input_size = ViewCopyConverter._safe_compute_flat_size(list(x.size()))
        flat_output_size = ViewCopyConverter._safe_compute_flat_size(list(y.size()))

        if tensor_rank(y) >= 8 or flat_input_size != flat_output_size:
            return False

        return True

    @classmethod
    def supports_partitioning_result(
        cls,
        node: Node,
        partition_list: list[Partition],
        custom_delegation_options: CustomDelegationOptions,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
    ):
        view_copy_partitions = [
            partition for partition in partition_list if node in partition.nodes
        ]
        assert len(view_copy_partitions) == 1
        non_q_dq_partition_nodes = list(
            filter(is_not_qdq_node, view_copy_partitions[0].nodes)
        )

        if len(non_q_dq_partition_nodes) == 1:
            # The `view_copy` cannot be the only node in a partition.
            return False

        input_format = node.args[0].meta[NXP_NODE_FORMAT]
        output_format = node.meta[NXP_NODE_FORMAT]
        input_shape = list(node.args[0].meta["val"].shape)
        output_shape = list(node.meta["val"].shape)
        to_nchw_perm = create_channels_last_to_channels_first_permutation(
            len(input_shape), True
        )
        to_nhwc_perm = create_channels_first_to_channels_last_permutation(
            len(output_shape), True
        )
        channels_last_input_shape = apply_permutation_to(
            input_shape,
            create_channels_first_to_channels_last_permutation(len(input_shape), True),
        )

        if input_format.is_channels_first() and (not output_format.is_channels_first()):
            # The `view_copy` removes node format. Conversion will require the addition of a `Transpose` operator.
            # Make sure the `Transpose` will be supported.

            if not transposition_is_supported_on_neutron(
                channels_last_input_shape, to_nchw_perm, neutron_target_spec
            ):
                # The `Transpose` would have to be removed by the `PermuteFullyConnectedWeightsAfterReshape` pass.
                # Make sure it will be applied.
                users = get_non_qdq_users(node)
                if len(users) != 1 or (linear_node := users[0]).target not in [
                    exir_ops.edge.aten.addmm.default,
                    exir_ops.edge.aten.mm.default,
                ]:
                    return False

                if linear_node not in view_copy_partitions[0].nodes:
                    # The `mm` / `addmm` node will not be delegated within this partition.
                    return False

                # Make sure the specific requirements of the `PermuteFullyConnectedWeightsAfterReshape` are satisfied.
                weights_index = (
                    2 if linear_node.target == exir_ops.edge.aten.addmm.default else 1
                )
                if not (
                    input_shape[0] == output_shape[0]  # Preserve batch.
                    and len(output_shape) == 2
                    and output_shape[1]
                    == linear_node.args[weights_index].meta["val"].shape[0]
                ):
                    return False

        elif (
            not input_format.is_channels_first()
        ) and output_format.is_channels_first():
            # The `view_copy` introduces node format. Conversion will require the addition of a `Transpose` operator.
            # Make sure the `Transpose` will be supported.
            if not transposition_is_supported_on_neutron(
                output_shape, to_nhwc_perm, neutron_target_spec
            ):
                return False

        elif input_format.is_channels_first() and output_format.is_channels_first():
            # The `view_copy` works with the channels first format, so both tensors will end up being transposed.
            # Make sure these transpositions are supported.
            if not (
                transposition_is_supported_on_neutron(
                    channels_last_input_shape, to_nchw_perm, neutron_target_spec
                )
                and transposition_is_supported_on_neutron(
                    output_shape, to_nhwc_perm, neutron_target_spec
                )
            ):
                return False

        return True

    @staticmethod
    def _safe_compute_flat_size(shape: list[int | str]) -> int:
        """Compute the flat size of a tensor with given shape. Strings and negative dimensions are treated as '1'.

        :param shape: Shape of the tensor. Can include integers and strings.
        :return: The flat size of the tensor.
        """
        flat_size = 1
        for dim in shape:
            if isinstance(dim, int) and dim > 1:
                flat_size *= dim

        return flat_size

    def convert(self, node: Node):
        """Convert the `aten.view_copy` operator to TFLite `Reshape`."""
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        ops = OpsList(middle_op=t_op)

        if (
            x.quantization is not None
            and y.quantization is None
            and "cluster" in node.meta
        ):
            # We know this node is part of QDQ cluster, so we can propagate quantization to inputs of "call_function"
            # node of this cluster.
            quantization_utils.propagate_quantization(x, y)

            y.type = x.type
            assert x.quantization == y.quantization, (
                "ViewCopyConverter: Q-params of input and output doesn't match. This "
                "indicates error in quantizer."
            )

        new_shape = ensure_reshape_transposition(self.builder, ops)

        # Create the TFLite Reshape with the new shape
        t_op.builtin_options = reshape_options.Reshape(new_shape)

        # Required by neutron-converter, but it will remove this tensor in optimization phase
        new_shape_tensor = self.builder.create_tensor_for_data(
            np.asarray(new_shape, dtype=np.int32), "new_shape"
        )
        t_op.tmp_inputs.append(new_shape_tensor)

        self.builder.append_operators(ops.flatten())
