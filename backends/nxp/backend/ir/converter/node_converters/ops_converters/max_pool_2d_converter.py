# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np

from executorch.backends.nxp.backend.ir.converter.conversion import (
    aten_translator,
    common,
)
from executorch.backends.nxp.backend.ir.converter.conversion.common import OpsList
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.lib.tflite.TensorType import TensorType
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    max_pool_2d_options,
)
from torch.fx import Node
from torch.nn import Parameter


class MaxPool2dConverter(NodeConverter):
    """Convert 'max_pool2d' operator to TFLite 'MaxPool2D'.
    NOTE: max_pool2d_with_indices is a different operator and is unsupported.
    """

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        n_args = len(node.args)

        dilation = node.args[4] if n_args >= 5 else [1, 1]
        ceil_mode = node.args[5] if n_args == 6 else False

        if any(dil != 1 for dil in dilation) or ceil_mode:
            return False

        if not NodeConverter._has_shared_q_params_if_quantized(node):
            return False

        return True

    def _get_pad_constant_value(self, input_type: TensorType) -> np.ndarray:
        """
        Get scalar NumPy array with constant value used as constant value for 'Pad' operator.

        :param input_type: Input tensor type.
        :return: Scalar array with single minimum value of given type.
        """

        match input_type:
            case TensorType.INT8:
                return np.asarray([np.iinfo(np.int8).min], dtype=np.int8)
            case TensorType.UINT8:
                return np.asarray([np.iinfo(np.uint8).min], dtype=np.uint8)
            case TensorType.FLOAT32:
                return np.asarray([np.finfo(np.float32).min], dtype=np.float32)
            case _:
                raise RuntimeError("Unexpected input type for MaxPool operator.")

    # noinspection PyMethodMayBeStatic
    def _convert_2d_max_pool(
        self, kernel_size, stride, padding, t_op: tflite_model.Operator
    ) -> list[tflite_model.Operator]:
        x = t_op.tmp_inputs[0]

        ops = OpsList(middle_op=t_op)
        t_op.builtin_options = max_pool_2d_options.MaxPool2D()
        t_op.builtin_options.filter_h = kernel_size[0]
        t_op.builtin_options.filter_w = kernel_size[1]
        common.assign_2d_strides(t_op.builtin_options, stride)
        t_op.builtin_options.padding, explicit_padding = (
            aten_translator.convert_padding(padding)
        )

        if explicit_padding is not None:
            # Need to prepend a 'Pad' operator, which adds min values for type.
            constant_value = self._get_pad_constant_value(x.type)
            pre_pad_op = self.builder.create_pad_operator_before(
                t_op, 0, explicit_padding, constant_value=constant_value
            )
            ops.add_pre(pre_pad_op)

        return ops.flatten()

    # Maxpool2d Node format: (Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False)
    def convert(self, node: Node):
        self.assert_convertible(node)

        n_args = len(node.args)

        kernel_size = node.args[1]
        stride = node.args[2]
        padding = node.args[3] if n_args >= 4 else [0, 0]

        t_op = self._create_tflite_op_with_io_tensors(node)
        ops_to_add = self._convert_2d_max_pool(kernel_size, stride, padding, t_op)
        self.builder.append_operators(ops_to_add)
