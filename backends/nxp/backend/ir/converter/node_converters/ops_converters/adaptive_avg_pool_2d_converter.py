# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging

import executorch.backends.nxp.backend.ir.lib.tflite.Padding as tflPadding
import torch

from executorch.backends.nxp.backend.data_format import NXP_NODE_FORMAT
from executorch.backends.nxp.backend.ir.converter.conversion import common
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    average_pool_2d_options,
)

from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from torch.fx import Node
from torch.nn import Parameter

KernelSize = tuple[int, int]
Stride = tuple[int, int]


class AdaptiveAvgPool2dConverter(NodeConverter):

    @staticmethod
    def _get_equivalent_avg_pool_parameters(node: Node) -> tuple[KernelSize, Stride]:
        input_size = node.args[0].meta["val"].shape[2:]  # Spatial dims from NCHW shape.
        output_size = node.args[1]
        stride = (input_size[0] // output_size[0], input_size[1] // output_size[1])
        kernel_size = (
            input_size[0] - (output_size[0] - 1) * stride[0],
            input_size[1] - (output_size[1] - 1) * stride[1],
        )

        return kernel_size, stride

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        if (
            format_ := node.meta.get(NXP_NODE_FORMAT)
        ) is None or not format_.is_channels_first():
            logging.warning(
                "NXP backend: `adaptive_avg_pool_2d` doesn't have the required input format for delegation. "
                "Please run `NodeFormatInference.identify_node_formats()` during lowering or report this issue."
            )
            return False

        input_size = node.args[0].meta["val"].shape
        output_size = node.args[1]

        if (input_size[-1] % output_size[-1] != 0) or (
            input_size[-2] % output_size[-2] != 0
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
        kernel_size, stride = (
            AdaptiveAvgPool2dConverter._get_equivalent_avg_pool_parameters(node)
        )

        if not NodeConverter.uses_quantization_type_for_io(
            node,
            supported_types=[torch.int8, torch.uint8],
            input_indices=[0],
            output_indices=[0],
        ):
            return False

        if any(k > 4096 for k in kernel_size):
            return False

        if any(s > 4096 for s in stride):
            return False

        return True

    def convert(self, node: Node):
        """Convert the '_adaptive_avg_pool2d' operator to NeutronIR 'AveragePool2D'.
        The ExecuTorch schema is:
            _adaptive_avg_pool2d(
                Tensor self,
                SymInt[2] output_size
            ) -> Tensor
        """
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)
        t_op.builtin_options = average_pool_2d_options.AveragePool2D()

        kernel_size, stride = self._get_equivalent_avg_pool_parameters(node)

        common.assign_2d_strides(t_op.builtin_options, stride)
        t_op.builtin_options.filter_h, t_op.builtin_options.filter_w = kernel_size
        t_op.builtin_options.padding = tflPadding.Padding.VALID

        self.builder.append_operators([t_op])
