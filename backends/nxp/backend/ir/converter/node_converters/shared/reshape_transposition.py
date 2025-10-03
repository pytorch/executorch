# Copyright 2023-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum

import numpy as np

from executorch.backends.nxp.backend.ir.converter.conversion import translator
from executorch.backends.nxp.backend.ir.converter.conversion.common import OpsList
from executorch.backends.nxp.backend.ir.tensor_formatting import TensorFormat


class SingleUnitaryDimensionChangeType(Enum):
    SQUEEZE = (0,)  # Removing one dimension with value 1
    UNSQUEEZE = 1  # Adding one dimensions with value 1


def _single_unitary_dimension_change(  # noqa C901
    from_shape, to_shape
) -> tuple[int, SingleUnitaryDimensionChangeType] | None:
    """
    Get change details (index of change and type of change) if there's only single unitary change
    between input shapes. If there is no such a change, None is returned otherwise.

    :param from_shape: First compared shape.
    :param to_shape: Second compared shape.
    :return: Tuple with change details (changed index and type of change) or None.
    """
    change_type = SingleUnitaryDimensionChangeType.UNSQUEEZE

    if (
        abs(len(from_shape) - len(to_shape)) != 1
    ):  # More than one added/removed dimension
        return None
    elif len(from_shape) > len(to_shape):  # Make sure 'from_shape' is a shorter one
        from_shape, to_shape = to_shape, from_shape
        change_type = SingleUnitaryDimensionChangeType.SQUEEZE

    # All dimensions in both shapes are ones
    if np.all(np.array(to_shape) == 1) and np.all(np.array(from_shape) == 1):
        return 0, change_type

    # Iterate from the beginning of the shorter shape and find first non-matching dimension
    first_non_matching_forward = None
    for i in range(len(from_shape)):
        if from_shape[i] != to_shape[i]:
            first_non_matching_forward = i
            break

    # Iterate from the end of the shorter shape and find first non-matching dimension
    first_non_matching_backward = None
    for i in range(-1, -len(from_shape) - 1, -1):
        if from_shape[i] != to_shape[i]:
            first_non_matching_backward = i
            break

    # Normalize (from negative to positive value) index of non-matching dimension with
    # respect to shape with more dims
    if first_non_matching_backward is not None:
        first_non_matching_backward = first_non_matching_backward + len(to_shape)

    # 'from_shape' completely matched the beginning of 'to_shape', for example:
    # from_shape=(2,3,4), to_shape=(2,3,4,1)
    if first_non_matching_forward is None and first_non_matching_backward is not None:
        if to_shape[first_non_matching_backward] == 1:
            return first_non_matching_backward, change_type
    # 'from_shape' completely matched the end of 'to_shape', for example:
    # from_shape=(2,3,4), to_shape=(1,2,3,4)
    elif first_non_matching_forward is not None and first_non_matching_backward is None:
        if to_shape[first_non_matching_forward] == 1:
            return first_non_matching_forward, change_type
    # 'from_shape' matched partially from the beginning and partly from the end of 'to_shape',
    # for example: from_shape=(2,3,4), to_shape=(2,1,3,4)
    elif (first_non_matching_forward == first_non_matching_backward) and to_shape[
        first_non_matching_forward
    ] == 1:
        return first_non_matching_forward, change_type

    return None


def _get_permutation_for_single_unitary_change_in_NC_dims(
    shape_from: list[int], to_shape: list[int]
) -> list[int] | None:
    """
    Get permutation used by prepended 'Transpose' operator if there's only single unitary
    dimension change (single added/removed dimension with value 1) in batch or channel dimension
    done by 'Reshape' operator.

    :param shape_from: Input shape of 'Reshape' operator.
    :param to_shape: Output shape of 'Reshape' operator.
    :return: Permutation as list of ints, or None if there is no single unitary change in NC dimensions.
    """

    old_shape_channel_first = translator.dims_to_channels_first(shape_from)
    new_shape_channel_first = translator.dims_to_channels_first(to_shape)

    change_details = _single_unitary_dimension_change(
        old_shape_channel_first, new_shape_channel_first
    )

    # Mapping from dimension change details into permutation used in prepended 'Transpose' op
    # in format: permutation_mapping[SQUEEZE/UNSQUEEZE][old_shape dimension][changed index]
    permutation_mapping = {
        SingleUnitaryDimensionChangeType.SQUEEZE: {
            4: {
                0: [0, 3, 2, 1],
                1: [0, 2, 1, 3],
            },
            5: {
                0: [0, 4, 2, 3, 1],
                1: [0, 2, 3, 1, 4],
            },
        },
        SingleUnitaryDimensionChangeType.UNSQUEEZE: {
            3: {
                0: [2, 1, 0],
                1: [0, 2, 1],
            },
            4: {
                0: [3, 1, 2, 0],
                1: [0, 3, 1, 2],
            },
        },
    }

    if change_details is not None:
        changed_index, change_type = change_details
        if changed_index > 1:
            # There is single unitary change in other than NC dimensions -> ignoring
            return None
        return permutation_mapping[change_type][len(shape_from)][changed_index]

    return None


def ensure_reshape_transposition(builder, ops: OpsList) -> list[int]:
    """
    Ensure transposition of Reshape operator is defined correctly based on tensor format.
    New operators (Transpose) are added into "ops" collection when necessary.

    :param builder: ModelBuilder instance.
    :param ops: OpsList instance with Reshape as "middle_op".
    :return: New shape of Reshape operator.
    """
    t_op = ops.middle_op
    input_tensor = t_op.tmp_inputs[0]
    input_rank = input_tensor.rank
    input_format = input_tensor.tensor_format
    output_tensor = t_op.tmp_outputs[0]
    output_rank = output_tensor.rank
    output_format = output_tensor.tensor_format

    # Shapes in TFLite format
    input_shape = input_tensor.shape.vector
    new_shape = output_tensor.shape.vector

    if input_format.is_channels_last() and not output_format.is_channels_last():
        # The dimensions of the tensor lose their meaning! Insert a transpose op, to change input to match ExecuTorch.

        permutation = list(
            translator.create_channels_last_to_channels_first_permutation(input_rank)
        )
        transpose = builder.create_transpose_operator_before(t_op, 0, permutation)
        transpose.tmp_outputs[0].tensor_format = TensorFormat.CHANNELS_FIRST

        ops.add_pre(transpose)

    elif not input_format.is_channels_last() and output_format.is_channels_last():
        # The Reshape introduces format to the tensor (2D -> 4D  for example)
        # The `view_copy` outputs a 'channels first' tensor. This has to stay the same, and then a Transpose operator
        # must be added, to change the tensor to 'channels last'.

        permutation = list(
            translator.create_channels_first_to_channels_last_permutation(output_rank)
        )
        transpose = builder.create_transpose_operator_after(t_op, 0, permutation)
        transpose.tmp_inputs[0].tensor_format = TensorFormat.CHANNELS_FIRST

        new_shape = translator.dims_to_channels_first(new_shape)

        ops.post_ops.insert(0, transpose)
    elif input_format.is_channels_last() and output_format.is_channels_last():
        batch_match = input_tensor.shape.vector[0] == output_tensor.shape.vector[0]
        channels_match = input_tensor.shape.vector[-1] == output_tensor.shape.vector[-1]

        if batch_match and channels_match:
            # It is safe to skip 'Transposition' at all because 'NC' dimensions are the same and
            # not mixed with other dimensions
            pass
        elif permutation := _get_permutation_for_single_unitary_change_in_NC_dims(
            input_shape, new_shape
        ):
            # Single added/removed dimension with value 1
            transpose = builder.create_transpose_operator_before(t_op, 0, permutation)
            transpose.tmp_outputs[0].tensor_format = (
                TensorFormat.RESHAPE_SINGLE_UNITARY_TRANSPOSITION
            )

            ops.add_pre(transpose)
        else:
            # The only way to convert this correctly is to insert a Transpose operator before, to make the input
            # channels first, and another Transpose after, to make the output channels last again.
            last_to_first_perm = (
                translator.create_channels_last_to_channels_first_permutation(
                    input_rank
                )
            )
            ops.add_pre(
                builder.create_transpose_operator_before(
                    t_op, 0, list(last_to_first_perm)
                )
            )
            t_op.tmp_inputs[0].tensor_format = TensorFormat.CHANNELS_FIRST

            new_shape = translator.dims_to_channels_first(new_shape)

            first_to_last_perm = (
                translator.create_channels_first_to_channels_last_permutation(
                    output_rank
                )
            )
            ops.post_ops.insert(
                0,
                builder.create_transpose_operator_after(
                    t_op, 0, list(first_to_last_perm)
                ),
            )
            t_op.tmp_outputs[0].tensor_format = TensorFormat.CHANNELS_FIRST

    return new_shape
