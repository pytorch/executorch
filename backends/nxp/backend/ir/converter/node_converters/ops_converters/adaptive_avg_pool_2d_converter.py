# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.nxp.backend.ir.lib.tflite.Padding as tflPadding
from executorch.backends.nxp.backend.ir.converter.conversion import common
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    average_pool_2d_options,
)
from torch import Size
from torch.fx import Node
from torch.nn import Parameter


class AdaptiveAvgPool2dConverter(NodeConverter):

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        input_size = node.args[0].meta["val"].shape
        output_size = node.args[1]

        if (input_size[-1] % output_size[-1] != 0) or (
            input_size[-2] % output_size[-2] != 0
        ):
            return False

        if not NodeConverter._has_shared_q_params_if_quantized(node):
            return False

        return True

    # noinspection PyMethodMayBeStatic
    def _convert_adaptive_avg_pool_2d(
        self, input_size: Size, output_size: list[int], t_op: tflite_model.Operator
    ):
        t_op.builtin_options = average_pool_2d_options.AveragePool2D()
        stride = [input_size[-2] // output_size[-2], input_size[-1] // output_size[-1]]
        common.assign_2d_strides(t_op.builtin_options, stride)
        t_op.builtin_options.filter_h = (
            input_size[-2] - (output_size[-2] - 1) * stride[-2]
        )
        t_op.builtin_options.filter_w = (
            input_size[-1] - (output_size[-1] - 1) * stride[-1]
        )
        t_op.builtin_options.padding = tflPadding.Padding.VALID

    # AdaptiveAvgPool2d Node format: (Tensor self, SymInt[2] output_size)
    def convert(self, node: Node):
        """Convert '_adaptive_avg_pool2d' operator to TFLite 'AveragePool2D'."""
        self.assert_convertible(node)

        input_size = node.args[0].meta["val"].shape
        output_size = node.args[1]

        t_op = self._create_tflite_op_with_io_tensors(node)

        self._convert_adaptive_avg_pool_2d(input_size, output_size, t_op)
        self.builder.append_operators([t_op])
