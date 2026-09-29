# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
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
    node: Node,
) -> bool:
    """This function determines if the current NeutronSoftware properly supports an activation operator represented by the given node.

    :param node: The node representing the activation operator.
    """
    # Prevent circular import
    from executorch.backends.nxp.backend.ir.converter.node_converter import (
        NodeConverter,
    )

    return NodeConverter.uses_quantization_type_for_io(
        node,
        supported_types=[torch.int8, torch.uint8],
        input_indices=[0],
        output_indices=[0],
    )
