# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.backend.ir import logger
from executorch.backends.nxp.backend.ir.converter.builder import model_builder
from executorch.backends.nxp.backend.ir.converter.conversion import translator
from executorch.backends.nxp.backend.ir.converter.conversion.common import (
    OpsList,
    try_get_input,
)
from executorch.backends.nxp.backend.ir.converter.tensor_utils import tensor_has_data
from executorch.backends.nxp.backend.ir.lib.tflite.ActivationFunctionType import (
    ActivationFunctionType,
)
from executorch.backends.nxp.backend.ir.tensor_formatting import TensorFormat
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model


def ensure_correct_tensor_formatting(
    t_op: tflite_model.Operator, builder: model_builder.ModelBuilder, ops: OpsList
):
    """Make sure that all input and output tensors of 't_op' have the correct format. 't_op' is assumed to be an LSTM
         or RNN operator.

        The LSTM/RNN may be using channels last tensors, because of the surrounding operators. LSTM/RNN requires its own
         format, however I think the input tensors should be marked as 'FORMATLESS', because the main inputs of TFLite
         and ONNX version of the operators have the same shape.
        I believe that the cleanest and most robust way to solve this, is to mark LSTM/RNN as an operator which can
         change the formats of its tensors, and solve any format related issues in this module.

    :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX LSTM/RNN operator.
    :param builder: ModelBuilder object.
    :param ops: OpsList object, with operators to add to the model. May already contain some operators.
    """

    if t_op.tmp_inputs[0].tensor_format == TensorFormat.FORMATLESS:
        # Nothing to be done. All tensors should be formatless.
        return

    # Permute the inputs.
    for idx, tensor in enumerate(t_op.tmp_inputs.copy()):
        if tensor.tensor_format.is_channels_last():
            revert_perm = translator.create_channels_last_to_channels_first_permutation(
                tensor.rank, return_list=True
            )
            if tensor_has_data(tensor):
                translator.permute_static_tensor(tensor, revert_perm)

            else:
                # Prepend a Transpose operator.
                transpose = builder.create_transpose_operator_before(
                    t_op, idx, revert_perm
                )
                ops.pre_ops.append(transpose)

            t_op.tmp_inputs[idx].tensor_format = TensorFormat.FORMATLESS

    # LSTM/RNN produces 'FORMATLESS' outputs. However, if the output tensors have the 'channels_last' format, Transpose
    #  operators must be added, to actually make the inputs 'channels_last'.
    for idx, tensor in enumerate(t_op.tmp_outputs.copy()):
        if tensor.tensor_format.is_channels_last():
            # Append a Transpose operator.
            revert_perm = translator.create_channels_first_to_channels_last_permutation(
                tensor.rank, return_list=True
            )
            transpose = builder.create_transpose_operator_after(t_op, idx, revert_perm)
            ops.post_ops.append(transpose)

            t_op.tmp_outputs[idx].tensor_format = TensorFormat.FORMATLESS


def get_activation_function_for_name(
    name: str, op_type: str = "LSTM"
) -> ActivationFunctionType:
    get_activation_function_for_name.map = {
        "Tanh": ActivationFunctionType.TANH,
        "Relu": ActivationFunctionType.RELU,
    }

    if act_fun := get_activation_function_for_name.map.get(name, None):
        return act_fun

    # Couldn't find a corresponding activation function
    logger.e(
        logger.Code.CONVERSION_IMPOSSIBLE,
        f"Conversion of ONNX {op_type} with activation function '{name}' is not possible.",
    )


def check_sequence_lens(
    t_op: tflite_model.Operator, seq_length: int, op_type: str = "LSTM"
):
    """Check if the 'sequence_lens' operand of ONNX LSTM/RNN has an effect. If it does, exit with error.

    :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator.
    :param seq_length: The first dimension of the main LSTM input.
    :param op_type: Operator type of 't_op'. Used only for printing a specific error message.
    """
    if sequence_lens := try_get_input(t_op, 4):
        # 'sequence_lens' allows each sequence to have a different length. As far as I can tell, TFLite doesn't support
        #  this.
        if (not tensor_has_data(sequence_lens)) or any(
            elt != seq_length for elt in sequence_lens.tmp_buffer.data
        ):
            # The 'sequence_lens' is either dynamic, or static with at least one value different from 'seq_length'.
            # Conversion most likely impossible.
            logger.e(
                logger.Code.CONVERSION_IMPOSSIBLE,
                f"Conversion of ONNX {op_type} with 'sequence_lens' input is not possible.",
            )
