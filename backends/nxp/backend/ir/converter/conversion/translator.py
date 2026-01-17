# Copyright 2023 Martin Pavella
# Copyright 2023-2025 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    translator

Module contains functions for context-free conversion of various
things from ExecuTorch to NeutronIR.
"""

from typing import Any, Collection, List, Optional, Sequence

import executorch.backends.nxp.backend.ir.lib.tflite.Padding as tflPadding
import executorch.backends.nxp.backend.ir.logger as logger
import executorch.backends.nxp.backend.ir.tflite_generator.tflite_model as tflite_model

import numpy as np
import torch
from executorch.backends.nxp.backend.ir.lib.tflite.TensorType import TensorType


def permute_static_tensor(tensor: tflite_model.Tensor, perm: list[int]):
    """Take a static NeutronIR tensor and permute its shape and data according to the permutation in 'perm'.

    :param tensor: Static NeutronIR tensor to permute.
    :param perm: Permutation to apply to the tensor.
    """

    logger.internal_assert(
        tensor.tmp_buffer is not None, "permute_static_tensor: tensor is not static."
    )

    data = tensor.tmp_buffer.data
    data = np.transpose(data, perm)

    shape = apply_permutation_to(tensor.shape.vector.copy(), perm)
    logger.internal_assert(
        shape == list(data.shape), "permute_static_tensor: shapes do not match."
    )

    tensor.tmp_buffer.data = data
    tensor.shape = tflite_model.Shape(shape)


def get_tflite_tensor_shape_with_explicit_padding(
    tflite_shape: List[int], explicit_padding: List[List[int]]
) -> List[int]:
    """Get the resulting shape of a tensor with shape 'tflite_shape' (in NeutronIR format), after 'explicit_padding' is
    applied to it.
    """

    if (len(tflite_shape) != len(explicit_padding)) or any(
        len(sub_list) != 2 for sub_list in explicit_padding
    ):
        logger.e(
            logger.Code.INTERNAL_ERROR,
            f"Cannot apply padding '{explicit_padding}' to NeutronIR shape '{tflite_shape}'!",
        )

    total_padding = [
        start + end for start, end in explicit_padding
    ]  # Total padding for each dimension

    padded_shape = []
    for dimension, padding in zip(tflite_shape, total_padding):
        if isinstance(dimension, int) and dimension > 0:
            padded_shape.append(dimension + padding)

        else:
            # Variable shape

            if padding == 0:
                padded_shape.append(dimension)

            else:
                # Cannot add padding to a variable dimension.
                logger.e(
                    logger.Code.CONVERSION_IMPOSSIBLE,
                    "Adding explicit padding to a variable sized tensor is not supported!",
                )

    return padded_shape


def dims_to_channels_first(channels_last_dimensions: List[int]) -> List[int]:
    """Convert a list of ints which represent dimensions in the channels last (NeutronIR) format to the channels first
    (ExecuTorch) format.
    """
    assert len(channels_last_dimensions) > 0, "Dimensions list is empty!"

    if len(channels_last_dimensions) == 1:
        return [0]

    res = list(channels_last_dimensions)

    res.insert(1, res.pop())  # Insert 'C' (last item) to index 1

    return res


def dims_to_channels_last(channels_first_dimensions: List[int]) -> List[int]:
    """Convert a list of ints which represent dimensions in the channels first (ExecuTorch) format to the channels last
    (NeutronIR) format.
    """
    assert len(channels_first_dimensions) > 0, "Dimensions list is empty!"

    if len(channels_first_dimensions) == 1:
        return [0]

    res = list(channels_first_dimensions)

    res.append(res.pop(1))  # Move 'C' (idx 1) to the end

    return res


def collections_equal(col_a, col_b):
    """Compare each individual element of both collections.
    They can be any combination of lists, tuples or numpy arrays.
    Return True if they are equal.
    """
    if len(col_a) != len(col_b):
        return False

    for a, b in zip(col_a, col_b):
        if a != b:
            return False
    return True


def _calculate_effective_kernel_shape(
    kernel_shape: List[int], dilations: Optional[List[int]]
) -> List[int]:
    """Calculate the reach of a kernel with respect to its shape and dilations.
    For example a [3, 3] kernel with dilations [2, 2] has effective shape of [5, 5].
    """

    if dilations is None:
        dilations = [1] * len(kernel_shape)

    return [(k - 1) * d + 1 for k, d in zip(kernel_shape, dilations)]


def _same_upper_equals_same_lower(
    tflite_input_shape: List[int],
    tflite_output_shape: List[int],
    o_kernel_shape: List[int],
    o_strides: Optional[List[int]] = None,
    o_dilations: Optional[List[int]] = None,
) -> bool:
    """Determine if in a given particular setting, the values of the ExecuTorch `auto_pads` attribute SAME_UPPER and
    SAME_LOWER represent the exact same padding.
    """

    padding, offset = tflite_compute_padding_with_offset(
        tflite_input_shape, o_kernel_shape, tflite_output_shape, o_strides, o_dilations
    )

    # Only if offset for every dimension is 0, SAME_UPPER and SAME_LOWER will behave equally.
    return all(elt == 0 for elt in offset)


def _tflite_padding_compute_output_size(
    padding: tflPadding.Padding,
    tflite_spatial_input_shape: List[int],
    tflite_kernel_shape: List[int],
    strides: Optional[List[int]] = None,
    dilations: Optional[List[int]] = None,
) -> List[int]:
    """
    Calculates the output shape of the tensor with particular setting as tflite would. Implementation corresponds to
    tensorflow/lite/kernels/padding.h:ComputeOutSize()
    :param padding: NeutronIR Padding value - 'Same' or 'Valid'
    :param tflite_spatial_input_shape: input tensor shape
    :param tflite_kernel_shape: convolution kernel shape
    :param strides: strides (default is 1)
    :param dilations: dilation (default is 1)
    :return: Output shape of the tensor with particular padding settings
    """
    if strides is None:
        strides = [1] * len(tflite_kernel_shape)

    effective_kernel_shape = _calculate_effective_kernel_shape(
        tflite_kernel_shape, dilations
    )

    if padding == tflPadding.Padding.SAME:
        return [
            (in_shape + stride - 1) // stride
            for in_shape, stride in zip(tflite_spatial_input_shape, strides)
        ]
    elif padding == tflPadding.Padding.VALID:
        return [
            (in_shape + stride - ef_kernel_shape) // stride
            for in_shape, stride, ef_kernel_shape in zip(
                tflite_spatial_input_shape, strides, effective_kernel_shape
            )
        ]


def tflite_compute_padding_with_offset(
    tflite_input_shape: List[int],
    tflite_kernel_shape: List[int],
    tflite_output_shape: List[int],
    strides: Optional[List[int]] = None,
    dilations: Optional[List[int]] = None,
) -> (List[int], List[int]):
    """
    Calculate padding and offset for each dimension for particular convolution setting as NeutronIR.
    Implementation corresponds to tensorflow/lite/kernels/padding.h:ComputePaddingWithOffset()
    :param tflite_input_shape: tensorflow lite input shape
    :param tflite_kernel_shape: tensorflow lite kernel shape
    :param tflite_output_shape: tensorflow lite output shape
    :param strides: stride setting, default is 1
    :param dilations: dilation setting, default is 1
    :return: (padding, offset) - padding and offset for each axis. Padding is added on beginning and end of the axis.
             Offset to be optionally added to end of the axis if odd.
    """
    if strides is None:
        strides = [1] * len(tflite_kernel_shape)

    spatial_input_shape = tflite_input_shape[1:-1]  # The spatial portion of the input
    spatial_output_shape = tflite_output_shape[
        1:-1
    ]  # The spatial portion of the output

    effective_kernel_shape = _calculate_effective_kernel_shape(
        tflite_kernel_shape, dilations
    )

    total_padding = [
        (spatial_output - 1) * stride + effective_kernel - spatial_input
        for spatial_output, stride, effective_kernel, spatial_input in zip(
            spatial_output_shape, strides, effective_kernel_shape, spatial_input_shape
        )
    ]

    padding = [tp // 2 for tp in total_padding]
    offset = [tp % 2 for tp in total_padding]

    return padding, offset


def _is_same_padding(
    o_pads: List[int],
    tflite_input_shape: List[int],
    tflite_output_shape: List[int],
    o_kernel_shape: List[int],
    o_strides: Optional[List[int]] = None,
    o_dilations: Optional[List[int]] = None,
) -> bool:
    """Determine if given ExecuTorch 'pads' padding can be represented exactly with the NeutronIR 'SAME' padding type.

    :param o_pads: ExecuTorch 'pads' attribute.
    :param tflite_input_shape: The shape of the main input of the operator in NeutronIR format.
    :param tflite_output_shape: The shape of the main output of the operator in NeutronIR format.
    :param o_kernel_shape: ExecuTorch 'kernel_shape' attribute.
    :param o_strides: ExecuTorch 'strides' attribute. Can be omitted.
    :param o_dilations: ExecuTorch 'dilations' attribute. Can be omitted.
    """

    if len(tflite_input_shape) == 0 or len(tflite_output_shape) == 0:
        logger.e(
            logger.Code.INVALID_TENSOR_SHAPE,
            f"Cannot verify that padding '{o_pads}' can be represented as 'SAME' for input shape "
            f"'{tflite_input_shape}' and output shape '{tflite_output_shape}'.",
        )

    # Calculate if the output shape corresponds to Same padding setting in NeutronIR
    tflite_spatial_input_shape = tflite_input_shape[1:-1]
    tmp_spatial_output_shape = _tflite_padding_compute_output_size(
        tflPadding.Padding.SAME,
        tflite_spatial_input_shape,
        o_kernel_shape,
        o_strides,
        o_dilations,
    )
    if tmp_spatial_output_shape != tflite_output_shape[1:-1]:
        return False

    # For every dimension, the padding is added to the start and end of the dimension.
    # NeutronIR padding 'SAME' tries to split it evenly, but in case of odd padding, 'SAME' adds the excess 1 at the end.
    # NeutronIR represents this in the offset. The offset is added to the end of particular dimension,
    # i.e. bottom for H dim, right for W dim and so on.
    # ExecuTorch represents this in 'pads' as [x1_begin, x2_begin,... , x1_end, x2_end,...].
    padding, offset = tflite_compute_padding_with_offset(
        tflite_input_shape, o_kernel_shape, tflite_output_shape, o_strides, o_dilations
    )
    start_padding = padding
    end_padding = [p + o for p, o in zip(padding, offset)]
    effective_padding = start_padding + end_padding

    if effective_padding != o_pads:
        return False

    return True


def combine_permutations(
    permutation1: Sequence[int], permutation2: Sequence[int]
) -> List[int]:
    """Combine 2 permutations into 1.

    :param permutation1: The first permutation to apply.
    :param permutation2:  The second permutation to apply.
    :return: The combined permutation.
    """
    if len(permutation1) != len(permutation2):
        logger.e(
            logger.Code.INTERNAL_ERROR,
            "translator.combine_permutations(): permutations have different size!",
        )

    return [permutation1[perm2] for perm2 in permutation2]


def nhc_dimensions_to_nhwc(nhc_dimensions: List[int]) -> List[int]:
    """Convert a list of ints representing the shape of an NHC tensor to NHWC, where W = 1."""
    nhwc_dimensions = nhc_dimensions.copy()
    nhwc_dimensions.insert(2, 1)

    return nhwc_dimensions


def shape_from_numpy(numpy_array):
    """Return a 'Shape' object representing the shape of given 'numpy_array'."""
    dims = list(numpy_array.shape)
    return tflite_model.Shape(dims)


def executorch_explicit_padding_to_tflite(
    executorch_pads: list[int],
) -> list[list[int]]:
    """Convert the attribute or input 'pads' of the ExecuTorch 'Pad' operator to the 'paddings' input of the NeutronIR 'Pad'
     class of operators.

    This function does NOT take tensor formats into consideration.
    """

    start_padding = executorch_pads[
        : len(executorch_pads) // 2
    ]  # Padding at the start of each dimension
    end_padding = executorch_pads[
        len(executorch_pads) // 2 :
    ]  # Padding at the end of each dimension

    return list(zip(start_padding, end_padding))


def executorch_pads_to_tflite_explicit_padding(
    executorch_pads: List[int],
) -> List[List[int]]:
    """Convert an ExecuTorch attribute 'pads' of operators such as Conv, MaxPool or AveragePool, to a list of ints which is
    compatible with the NeutronIR 'Pad' operator.
    """

    tflite_padding = executorch_explicit_padding_to_tflite(executorch_pads)

    # NeutronIR also allows padding to the 'batch' and 'channels'. ExecuTorch does not
    tflite_padding.insert(0, [0, 0])
    tflite_padding.append([0, 0])

    return tflite_padding


def _get_explicit_tflite_padding_for_same_lower(
    tflite_input_shape: List[int],
    tflite_output_shape: List[int],
    o_kernel_shape: List[int],
    o_strides: Optional[List[int]] = None,
    o_dilations: Optional[List[int]] = None,
) -> List[List[int]]:
    """Get the NeutronIR explicit padding required to represent ExecuTorch 'SAME_LOWER' auto_pad for a particular setting.

    :param tflite_input_shape: NeutronIR (NHWC) shape of the input tensor of the operator.
    :param tflite_output_shape: NeutronIR (NHWC) shape of the output tensor of the operator.
    :param o_kernel_shape: ExecuTorch 'kernel_shape' attribute.
    :param o_strides: Optional ExecuTorch 'o_strides' attribute.
    :param o_dilations: Optional ExecuTorch 'o_dilations' attribute.

    :return: A NeutronIR style explicit padding, compatible with the NeutronIR 'Pad' operator.
    """

    padding, offset = tflite_compute_padding_with_offset(
        tflite_input_shape, o_kernel_shape, tflite_output_shape, o_strides, o_dilations
    )

    start_padding = [
        p + o for p, o in zip(padding, offset)
    ]  # In case of odd padding, the excess is added at the start
    end_padding = padding

    executorch_explicit_padding = start_padding + end_padding

    # Return explicit ExecuTorch padding converted to NeutronIR padding
    return executorch_pads_to_tflite_explicit_padding(executorch_explicit_padding)


def convert_data_to_channels_first(array: np.ndarray) -> np.ndarray:
    """Convert a numpy array representing the data of a tensor from the channels last format (NeutronIR), to channels
        first format (ExecuTorch).

    :param array: Numpy array holding the tensor's data.
    :return: The transformed data.
    """
    if len(array.shape) < 3:
        logger.e(
            logger.Code.INTERNAL_ERROR,
            f"translator.convert_data_to_channels_first(): 'array' only has '{len(array.shape)}' dimensions!",
        )

    return np.moveaxis(array, -1, 1)  # Move last axis (C), to index 1


def convert_data_to_channels_last(array: np.ndarray) -> np.ndarray:
    """Convert a numpy array representing the data of a tensor from the channels first format (ExecuTorch), to channels last
        format (NeutronIR).

    :param array: Numpy array holding the tensor's data.
    :return: The transformed data.
    """
    if len(array.shape) < 3:
        logger.e(
            logger.Code.INTERNAL_ERROR,
            f"translator.convert_data_to_channels_last(): 'array' only has '{len(array.shape)}' dimensions!",
        )

    return np.moveaxis(array, 1, -1)  # Move the second axis (C), to the end


def channels_last_shape_to_channels_first(
    nhwc_shape: tflite_model.Shape,
) -> tflite_model.Shape:
    """Create a channels first version of a channels last 'tflite_model.Shape' object."""

    dims = nhwc_shape.vector.copy()
    dims = dims_to_channels_first(dims)

    return tflite_model.Shape(dims)


def create_channels_last_to_channels_first_permutation(
    rank: int, return_list: bool = False
) -> np.ndarray | list[int]:
    """Return a numpy array with data that describes the permutation, which would change a tensor from the channels
    last (NeutronIR) format to the channels first (ExecuTorch) format.

    This permutation is compatible with the NeutronIR `Transpose` operator.

    :param rank: The rank of the required permutation.
    :param return_list: If True, the function returns a list of ints. If False, a numpy array is returned.
    :return: A numpy array, or a list of ints, representing the desired permutation.
    """

    perm = dims_to_channels_first(list(range(rank)))

    if return_list:
        return perm
    else:
        return np.asarray(perm, np.int32)


def create_channels_first_to_channels_last_permutation(
    rank: int, return_list: bool = False
) -> np.ndarray | list[int]:
    """Return a numpy array with data that describes the permutation, which would change a tensor from the channels
    first (ExecuTorch) format to the channels last (NeutronIR) format.

    This permutation is compatible with the NeutronIR `Transpose` operator.

    :param rank: The rank of the required permutation.
    :param return_list: If True, the function returns a list of ints. If False, a numpy array is returned.
    :return: A numpy array, or a list of ints, representing the desired permutation.
    """

    perm = dims_to_channels_last(list(range(rank)))

    if return_list:
        return perm
    else:
        return np.asarray(perm, np.int32)


def apply_permutation_to(target: List[Any], permutation: Collection[int]) -> List:
    """Permute a list according to a permutation. Uses the same permutation format as the NeutronIR Transpose operator.

    :param target: A list of any types, to permute. Must be same size as the permutation.
    :param permutation: The permutation to apply to the target.
    :return: Permuted list.
    """

    if len(target) != len(permutation):
        logger.e(
            logger.Code.INTERNAL_ERROR,
            "translator.apply_permutation_to(): 'target' and 'permutation' have different length!",
        )

    return [target[perm] for perm in permutation]


def create_inverse_permutation(permutation: List[int]) -> List[int]:
    """Create and return a permutation, that is the inverse of the given 'permutation' parameter.
        Uses the same permutation format as the NeutronIR Transpose operator.

    :param permutation: The permutation to create the inverse of.
    :return: Inverse permutation.
    """

    if set(permutation) != set(range(len(permutation))):
        # Irreversible permutation. For example [0, 1, 2, 2] (information is lost by applying permutation).
        logger.e(
            logger.Code.INTERNAL_ERROR,
            "translator.create_inverse_permutation(): permutation is not reversible!",
        )

    return [permutation.index(perm) for perm in range(len(permutation))]


def convert_data_type(torch_type: torch.TensorType) -> TensorType:
    """Convert Torch DataType to NeutronIR TensorType"""

    if torch_type == torch.float32:
        return TensorType.FLOAT32

    elif torch_type == torch.uint8:
        return TensorType.UINT8

    elif torch_type == torch.int8:
        return TensorType.INT8

    elif torch_type == torch.int32:
        return TensorType.INT32

    elif torch_type == torch.int64:
        return TensorType.INT64

    elif torch_type == torch.bool:
        return TensorType.BOOL

    else:
        logger.e(
            logger.Code.NOT_IMPLEMENTED,
            f"Conversion of Torch type '{torch_type}' not supported.",
        )


def torch_type_to_numpy_type(torch_type: torch.TensorType) -> np.ScalarType:
    """Convert Torch DataType to NeutronIR TensorType"""

    if torch_type == torch.float32:
        return np.dtype(np.float32)

    elif torch_type == torch.uint8:
        return np.dtype(np.uint8)

    elif torch_type == torch.int8:
        return np.dtype(np.int8)

    elif torch_type == torch.int32:
        return np.dtype(np.int32)

    elif torch_type == torch.int64:
        return np.dtype(np.int64)

    else:
        logger.e(
            logger.Code.NOT_IMPLEMENTED,
            f"Conversion of Torch type '{torch_type}' not supported.",
        )


def numpy_type_to_tf_lite(numpy_type: np.dtype) -> TensorType:  # noqa C901
    """Convert the numpy data type to a corresponding NeutronIR 'TensorType'.

    :param numpy_type: Numpy dtype to convert.
    :return: Corresponding NeutronIR TensorType.
    """
    numpy_type = numpy_type.type

    if numpy_type == np.float32:
        return TensorType.FLOAT32

    elif numpy_type == np.uint8:
        return TensorType.UINT8

    elif numpy_type == np.int8:
        return TensorType.INT8

    elif numpy_type == np.uint16:
        return TensorType.UINT16

    elif numpy_type == np.int16:
        return TensorType.INT16

    elif numpy_type == np.int32:
        return TensorType.INT32

    elif numpy_type == np.int64:
        return TensorType.INT64

    elif numpy_type == np.bytes_:
        return TensorType.STRING

    elif numpy_type == np.bool_:
        return TensorType.BOOL

    elif numpy_type == np.float16:
        return TensorType.FLOAT16

    elif numpy_type == np.float64:
        return TensorType.FLOAT64
    elif numpy_type == np.double:
        return TensorType.FLOAT64

    elif numpy_type == np.uint32:
        return TensorType.UINT32

    elif numpy_type == np.uint64:
        return TensorType.UINT64

    elif numpy_type == np.complex64:
        return TensorType.COMPLEX64

    elif numpy_type == np.complex128:
        return TensorType.COMPLEX128

    else:
        logger.e(
            logger.Code.CONVERSION_IMPOSSIBLE,
            f"Cannot convert numpy data type '{numpy_type}' to NeutronIR.",
        )


def tf_lite_type_to_numpy(tfl_type: TensorType) -> np.ScalarType:  # noqa C901
    """Convert NeutronIR TensorType to numpy dtype"""

    if tfl_type == TensorType.FLOAT32:
        return np.dtype(np.float32)

    elif tfl_type == TensorType.UINT8:
        return np.dtype(np.uint8)

    elif tfl_type == TensorType.INT8:
        return np.dtype(np.int8)

    elif tfl_type == TensorType.UINT16:
        return np.dtype(np.uint16)

    elif tfl_type == TensorType.INT16:
        return np.dtype(np.int16)

    elif tfl_type == TensorType.INT32:
        return np.dtype(np.int32)

    elif tfl_type == TensorType.INT64:
        return np.dtype(np.int64)

    elif tfl_type == TensorType.STRING:
        return np.dtype(np.bytes_)

    elif tfl_type == TensorType.BOOL:
        return np.dtype(np.bool_)

    elif tfl_type == TensorType.FLOAT16:
        return np.dtype(np.float16)

    elif tfl_type == TensorType.FLOAT64:
        return np.dtype(np.float64)

    elif tfl_type == TensorType.UINT32:
        return np.dtype(np.uint32)

    elif tfl_type == TensorType.UINT64:
        return np.dtype(np.uint64)

    elif tfl_type == TensorType.COMPLEX64:
        return np.dtype(np.complex64)

    elif tfl_type == TensorType.COMPLEX128:
        return np.dtype(np.complex128)

    else:
        logger.e(
            logger.Code.CONVERSION_IMPOSSIBLE,
            f"Cannot convert NeutronIR type '{tfl_type}' to numpy dtype.",
        )
