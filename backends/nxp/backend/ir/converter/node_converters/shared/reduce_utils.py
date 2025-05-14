# Copyright 2024 NXP
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
    """Create an `axes` tensor and assign it as an input to the `t_op`, which is expected to represent an ONNX
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


# def convert_axes_from_input_tensor(
#     t_op: tflite_model.Operator,
#     builder: ModelBuilder,
#     inspector: ONNXModelInspector,
#     ops: OpsList,
#     noop_with_empty_axes: int,
#     op_type: str,
# ):
#     """Verify the `axes` tensor (on input index 1) of the `t_op`, which is expected to represent an ONNX reduction
#     operator.
#     """
#     x = t_op.tmp_inputs[0]
#     rank = x.rank
#
#     if axes_tensor := try_get_input(t_op, 1):
#
#         # ONNX uses int64, while TFLite requires int32 for the `axes` tensor.
#         if axes_tensor.type != TensorType.INT64:
#             logger.e(
#                 logger.Code.INVALID_ONNX_OPERATOR,
#                 f"ONNX `{op_type}` has `axes` of type `{name_for_type(axes_tensor.type)}`, instead of INT64.",
#             )
#
#         # Try to get the inferred data for the `axes` input.
#         if (
#             axes_data := inspector.try_get_inferred_tensor_data(axes_tensor.name)
#         ) is not None:
#             # The `axes` were inferred during shape inference.
#             logger.d(
#                 f"Using inferred data for the `axes` input tensor of ONNX `{op_type}`."
#             )
#
#             # Create a new tensor, in case the original `axes` tensor is used by multiple ops.
#             axes_tensor = builder.create_tensor_for_data(
#                 axes_data.astype(np.int32), "axes"
#             )
#
#         # Make sure the `axes` are int32.
#         if tensor_has_data(axes_tensor):
#             # Cast the `axes` to int32 statically.
#             axes_tensor.tmp_buffer.data = axes_tensor.tmp_buffer.data.astype(np.int32)
#             axes_tensor.type = TensorType.INT32
#
#         else:
#             # The `axes` are dynamic and there is no inferred data for them. The shape inference is not possible in
#             #  this case, so it must have been skipped. If the `axes` are empty at runtime, ONNX will reduce over
#             #  all dimensions, whereas TFLite will not reduce at all. So the behavior is different, and it depends
#             #  on runtime data. Conversion could be implemented by adding multiple extra operators.
#             # I don't thing that completely prohibiting the conversion here is ideal, since the issue arises only in
#             #  an edge case, which is hopefully not very common. Just print a warning message for now.
#             logger.w(
#                 f"Conversion of ONNX `{op_type}` with a dynamic `axes` input will not be correct, if the `axes`"
#                 "are empty at runtime!"
#             )
#
#             # Insert a `Cast` op, to make the `axes` int32.
#             cast_op = builder.create_cast_before(t_op, 1, TensorType.INT32)
#             ops.add_pre(cast_op)
#
#             # For future references. Following code only cares about the final axes tensor.
#             axes_tensor = cast_op.tmp_outputs[0]
#
#         # Assign the new `axes_tensor` to the ReduceX operator.
#         t_op.tmp_inputs[1] = axes_tensor
#
#     else:
#         # No axes specified.
#
#         if noop_with_empty_axes == 1:
#             # ONNXRT: According to the documentation, the operator should do nothing in this situation. But that's
#             #  not what happens in ONNX Runtime. ORT seems to simply ignore the `noop_with_empty_axes` attribute.
#             #  https://github.com/microsoft/onnxruntime/issues/19147
#             # For now, exit with error. If later ORT adds support for this attribute, simply uncomment the
#             #  following code.
#
#             # if self.builder.operator_can_be_skipped(t_op, self.inspector):
#             #     # Skip the operator.
#             #     self.builder.redirect_tensor(t_op.tmp_outputs[0], t_op.tmp_inputs[0])
#             #     return []
#             #
#             # else:
#             #     # Return an operator which does nothing.
#             #     self.builder.turn_operator_to_identity(t_op)
#             #     return [t_op]
#
#             logger.e(
#                 logger.Code.INVALID_ONNX_OPERATOR,
#                 f"ONNX `{op_type}` has `noop_with_empty_axes` == 1 and the `axes` are not specified, which"
#                 " indicates that the operator should do nothing. This is however not supported by ONNX"
#                 " Runtime, and therefore the conversion is also not supported.",
#             )
#
#         else:
#             # Default is to reduce all axes.
#             axes_tensor = builder.create_tensor_for_data(
#                 np.arange(rank).astype(np.int32), "axes"
#             )
#
#             t_op.tmp_inputs[1:] = (
#                 []
#             )  # If the optional input was passed with name "", remove it.
#             t_op.tmp_inputs.append(axes_tensor)


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
        to_onnx_perm = translator.create_channels_last_to_channels_first_permutation(
            input_rank
        )
        to_tflite_perm = translator.create_channels_first_to_channels_last_permutation(
            output_rank, return_list=True
        )

        transpose_before = builder.create_transpose_operator_before(
            t_op, 0, to_onnx_perm
        )
        transpose_before.tmp_outputs[0].tensor_format = TensorFormat.CHANNELS_FIRST
        ops.add_pre(transpose_before)

        transpose_after = builder.create_transpose_operator_after(
            t_op, 0, to_tflite_perm
        )
        transpose_after.tmp_inputs[0].tensor_format = TensorFormat.CHANNELS_FIRST
        ops.post_ops.insert(0, transpose_after)

    elif input_format.is_channels_last() and not output_format.is_channels_last():
        # The dimensions of the tensor lose their meaning! Insert a transpose op, to change input to match ONNX.

        permutation = list(
            translator.create_channels_last_to_channels_first_permutation(input_rank)
        )
        transpose = builder.create_transpose_operator_before(t_op, 0, permutation)
        transpose.tmp_outputs[0].tensor_format = TensorFormat.CHANNELS_FIRST

        ops.add_pre(transpose)

    elif not input_format.is_channels_last() and output_format.is_channels_last():
        # The ReduceX introduces format to the tensor
        # The ONNX ReduceX outputs a 'channels first' tensor. This has to stay the same, and then a Transpose operator
        # must be added, to change the tensor to 'channels last'.

        permutation = list(
            translator.create_channels_first_to_channels_last_permutation(output_rank)
        )
        transpose = builder.create_transpose_operator_after(t_op, 0, permutation)
        transpose.tmp_inputs[0].tensor_format = TensorFormat.CHANNELS_FIRST

        ops.post_ops.insert(0, transpose)
