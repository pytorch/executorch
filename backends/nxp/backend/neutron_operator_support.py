# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec


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
    num_macs = neutron_target_spec.get_num_macs()

    if is_tensor_invariant_permutation(input_shape, permutation):
        # The `Transpose` will be turned into a `Reshape` by Neutron. The check includes the identity permutation.
        return True

    if permutation == [0, 3, 1, 2]:
        # NHWC -> NCHW
        n, h, w, c = input_shape

        if h * w * c % num_macs != 0:  # Official Neutron requirement.
            return False

        if not (
            c % num_macs == 0 and h * w % num_macs == 0
        ):  # Neutron would produce incorrect outputs.
            return False

        if n != 1:
            # Neutron only supports `Transpose` operators where the dimensions can be combined into 2 consecutive
            #  groups. These 2 groups are then transposed like a matrix, and the result is reshaped. Therefore, for the
            #  [0, 3, 1, 2] permutation, when h * w != 1 and c != 1, batch size must be 1.
            return False

        return True

    elif permutation == [0, 2, 3, 1]:
        # NCHW -> NHWC

        n, c, h, w = input_shape

        if w % num_macs != 0:  # Official Neutron requirement.
            return False

        if not (
            c % num_macs == 0 and h * w % num_macs == 0
        ):  # Neutron would produce incorrect outputs.
            return False

        if n != 1:
            # Neutron only supports `Transpose` operators where the dimensions can be combined into 2 consecutive
            #  groups. These 2 groups are then transposed like a matrix, and the result is reshaped. Therefore, for the
            #  [0, 2, 3, 1] permutation, when h * w != 1 and c != 1, batch size must be 1.
            return False

        return True

    return False
