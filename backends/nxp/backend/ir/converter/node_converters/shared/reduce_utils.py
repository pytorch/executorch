# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from executorch.backends.nxp.backend.ir.converter.builder.model_builder import (
    ModelBuilder,
)
from executorch.backends.nxp.backend.ir.converter.conversion import translator
from executorch.backends.nxp.backend.ir.converter.conversion.common import OpsList
from executorch.backends.nxp.backend.ir.tensor_formatting import TensorFormat
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model


def convert_axes_from_attribute(
    t_op: tflite_model.Operator, builder: ModelBuilder, axes: list[int] | None
):
    """Create an `axes` tensor and assign it as an input to the `t_op`, which is expected to represent an ExecuTorch
    reduction operator.
    """
    x = t_op.tmp_inputs[0]
    rank = x.rank

    if axes is None:
        # Default axes -> reduce over all dimensions.
        axes = np.arange(rank).astype(np.int32)

    else:
        # Axes are initialized.
        axes = np.asarray(axes, np.int32)

    # TFLite has `axes` as input tensor -> create it.
    axes_tensor = builder.create_tensor_for_data(axes, "axes")
    t_op.tmp_inputs.append(axes_tensor)


def ensure_reduce_transposition(builder, ops: OpsList):
    """
    Ensure transposition of ReduceX operator is defined correctly based on tensor format.
    New operators (Transpose) are added into "ops" collection when necessary.

    :param builder: ModelBuilder instance.
    :param ops: OpsList instance with operators related to currently converted ReduceX operator.
    """
    t_op = ops.middle_op
    input_tensor = t_op.tmp_inputs[0]
    input_rank = input_tensor.rank
    input_format = input_tensor.tensor_format
    output_tensor = t_op.tmp_outputs[0]
    output_rank = output_tensor.rank
    output_format = output_tensor.tensor_format

    if input_format.is_channels_last() and output_format.is_channels_last():
        to_executorch_perm = (
            translator.create_channels_last_to_channels_first_permutation(input_rank)
        )
        to_tflite_perm = translator.create_channels_first_to_channels_last_permutation(
            output_rank, return_list=True
        )

        transpose_before = builder.create_transpose_operator_before(
            t_op, 0, to_executorch_perm
        )
        transpose_before.tmp_outputs[0].tensor_format = TensorFormat.CHANNELS_FIRST
        ops.add_pre(transpose_before)

        transpose_after = builder.create_transpose_operator_after(
            t_op, 0, to_tflite_perm
        )
        transpose_after.tmp_inputs[0].tensor_format = TensorFormat.CHANNELS_FIRST
        ops.post_ops.insert(0, transpose_after)

    elif input_format.is_channels_last() and not output_format.is_channels_last():
        # The dimensions of the tensor lose their meaning! Insert a transpose op, to change input to match ExecuTorch.

        permutation = list(
            translator.create_channels_last_to_channels_first_permutation(input_rank)
        )
        transpose = builder.create_transpose_operator_before(t_op, 0, permutation)
        transpose.tmp_outputs[0].tensor_format = TensorFormat.CHANNELS_FIRST

        ops.add_pre(transpose)

    elif not input_format.is_channels_last() and output_format.is_channels_last():
        # The reduction operator introduces format to the tensor.
        # The ExecuTorch reduction operator outputs a 'channels first' tensor. This has to stay the same, and then a
        #  Transpose operator must be added, to change the tensor to 'channels last'.

        permutation = list(
            translator.create_channels_first_to_channels_last_permutation(output_rank)
        )
        transpose = builder.create_transpose_operator_after(t_op, 0, permutation)
        transpose.tmp_inputs[0].tensor_format = TensorFormat.CHANNELS_FIRST

        ops.post_ops.insert(0, transpose)
