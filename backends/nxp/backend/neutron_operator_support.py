# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.backend.data_format import NXP_NODE_FORMAT
from executorch.backends.nxp.backend.edge_helper import input_tensor
from executorch.backends.nxp.backend.ir.converter.conversion.translator import (
    dims_to_channels_last,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from torch.fx import Node


def is_tensor_invariant_permutation(
    input_shape: list[int], permutation: list[int]
) -> bool:
    def input_dim_is_not_one(index):
        return input_shape[index] != 1

    new_permutation = list(filter(input_dim_is_not_one, permutation))

    return new_permutation == sorted(new_permutation)


def transposition_is_supported_on_neutron(
    input_shape: list[int],
    permutation: list[int],
    neutron_target_spec: NeutronTargetSpec,
) -> bool:
    """This function determines if the current NeutronSoftware properly supports a `Transpose` operator with given
     `input_shape` and `permutation`.

    :param input_shape: The shape of the main input tensor of the `Transpose` operator.
    :param permutation: The permutation the `Transpose` operator is computing.
    :param neutron_target_spec: Object for querying the target platform to retrieve its properties.
    """
    # Neutron C currently supports all transpositions.
    # The function is not removed in case the support conditions ever change (for example with the introduction of
    #  Neutron S into ExecuTorch).
    return True


def activation_supported_on_target(
    node: Node, neutron_target_spec: NeutronTargetSpec
) -> bool:
    """This function determines if the current NeutronSoftware properly supports an activation operator represented by the given node.

    :param node: The node representing the activation operator.
    :param neutron_target_spec: Object for querying the target platform to retrieve its properties.
    """
    input_shape = list(input_tensor(node, 0).shape)
    if node.args[0].meta[NXP_NODE_FORMAT].is_channels_first():
        input_shape = dims_to_channels_last(input_shape)

    c = input_shape[-1]
    num_macs = neutron_target_spec.get_num_macs()

    # activations in Neutron are delegable only
    # if `num_channels` % `num_macs` == 0
    return c % num_macs == 0
