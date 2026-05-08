# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from executorch.backends.nxp.backend.ir.converter.conversion import (
    aten_translator,
    common,
)
from executorch.backends.nxp.backend.ir.converter.conversion.common import OpsList
from executorch.backends.nxp.backend.ir.converter.conversion.translator import (
    tf_lite_type_to_numpy,
)
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    average_pool_2d_options,
)

from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from torch.fx import Node
from torch.nn import Parameter


class AvgPool2dConverter(NodeConverter):

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        n_args = len(node.args)

        padding = node.args[3] if n_args >= 4 else [0, 0]
        ceil_mode = node.args[4] if n_args >= 5 else False
        count_include_pad = node.args[5] if n_args >= 6 else True
        divisor_override = node.args[6] if n_args == 7 else None
        _, explicit_padding = aten_translator.convert_padding(padding)

        if (
            (not count_include_pad and explicit_padding is not None)
            or divisor_override is not None
            or ceil_mode
        ):
            return False

        if not NodeConverter._has_shared_q_params_if_quantized(node):
            return False

        return True

    @staticmethod
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        kernel = node.args[1]
        stride = node.args[2]

        if custom_delegation_options.use_new_flow_neutron_c:
            # Requirements specified by the new Neutron flow documentation.

            supported_types = [torch.int8, torch.uint8]
            if not NodeConverter.uses_quantization_type_for_io(
                node, supported_types, [0]
            ):
                return False

            if any(k > 4096 for k in kernel):
                return False

            if any(s > 4096 for s in stride):
                return False

        return True

    # noinspection PyMethodMayBeStatic
    def _convert_2d_avg_pool(
        self, kernel_size, stride, padding, t_op: tflite_model.Operator
    ) -> list[tflite_model.Operator]:
        ops = OpsList(middle_op=t_op)
        t_op.builtin_options = average_pool_2d_options.AveragePool2D()
        t_op.builtin_options.filter_h = kernel_size[0]
        t_op.builtin_options.filter_w = kernel_size[1]
        common.assign_2d_strides(t_op.builtin_options, stride)
        t_op.builtin_options.padding, explicit_padding = (
            aten_translator.convert_padding(padding)
        )

        if explicit_padding is not None:
            # Need to prepend a 'Pad' operator, which adds 0s (or `zero_point` for the quantized case). But these will
            #  be included in the computation!
            input_quantization = t_op.tmp_inputs[0].quantization
            pad_value = (
                None
                if input_quantization is None
                else np.array(input_quantization.zero_point[0]).astype(
                    tf_lite_type_to_numpy(t_op.tmp_inputs[0].type)
                )
            )
            ops.add_pre(
                self.builder.create_pad_operator_before(
                    t_op, 0, explicit_padding, pad_value
                )
            )

        return ops.flatten()

    def convert(self, node: Node):
        """Convert the 'aten.avg_pool2d' operator to NeutronIR 'AveragePool2D'.
        The ExecuTorch schema is:
            aten.avg_pool2d(
                Tensor self,
                int[2] kernel_size,
                int[2] stride=[],
                int[2] padding=0,
                bool ceil_mode=False
                bool count_include_pad=True,
                int? divisor_override=None
            )
        """
        self.assert_convertible(node)

        kernel_size = node.args[1]
        stride = node.args[2]
        padding = node.args[3] if len(node.args) >= 4 else [0, 0]

        t_op = self._create_tflite_op_with_io_tensors(node)

        ops_to_add = self._convert_2d_avg_pool(kernel_size, stride, padding, t_op)
        self.builder.append_operators(ops_to_add)
