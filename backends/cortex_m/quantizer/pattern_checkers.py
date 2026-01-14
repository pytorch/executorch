# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from executorch.backends.cortex_m.passes.passes_utils import (
    is_channel_broadcast,
    is_channels_last,
)
from executorch.backends.cortex_m.quantizer.quantization_configs import (
    CMSIS_SOFTMAX_SCALE,
    CMSIS_SOFTMAX_ZERO_POINT,
)
from torch.fx import Node
from torchao.quantization.pt2e.quantizer import QuantizationSpec


class PatternCheck:
    """
    Base class for pattern checks.

    PatternChecks are used to define which which patterns are supported for quantization.
    For example, ADD in the Cortex-M backend does not support general broadcasting, so
    a PatternCheck can be used to filter out such patterns. They also only support per
    tensor quantization, so the PatternCheck filters out quantization configs that use
    per channel quantization.
    """

    @classmethod
    def is_per_tensor(cls, qspec: QuantizationSpec) -> bool:
        """
        Returns true if the given quantization spec is per-tensor, otherwise false.
        """
        return qspec.qscheme in (torch.per_tensor_affine, torch.per_tensor_symmetric)

    @classmethod
    def is_per_channel(cls, qspec: QuantizationSpec) -> bool:
        """
        Returns true if the given quantization spec is per-channel, otherwise false.
        """
        return qspec.qscheme in (torch.per_channel_affine, torch.per_channel_symmetric)

    @classmethod
    def check_pattern(cls, pattern: list[Node]) -> bool:
        """
        Returns true if the given pattern is supported, otherwise false.
        """
        return True

    @classmethod
    def check_quantization_config(cls, quantization_config: QuantizationConfig) -> bool:
        """
        Returns true if the given quantization config is supported, otherwise false.
        """
        return True


class CortexMAddMulCheck(PatternCheck):

    @classmethod
    def check_pattern(cls, pattern):
        """
        Checks that the pattern does not perform unsupported broadcasting.
        """
        for node in pattern:
            if len(node.all_input_nodes) == 2:
                t1 = get_first_fake_tensor(node.all_input_nodes[0])
                t2 = get_first_fake_tensor(node.all_input_nodes[1])
                if t1.shape != t2.shape and not (
                    is_channel_broadcast(t1, t2) and is_channels_last(t1)
                ):
                    return False

        return True

    @classmethod
    def check_quantization_config(cls, quantization_config):
        """
        Checks that the quantization config uses per-tensor int8 quantization.
        """
        is_per_tensor = PatternCheck.is_per_tensor(
            quantization_config.input_activation
        ) and PatternCheck.is_per_tensor(quantization_config.output_activation)
        is_int8 = (
            quantization_config.input_activation.dtype == torch.int8
            and quantization_config.output_activation.dtype == torch.int8
        )
        return is_per_tensor and is_int8


class CortexMConv2DCheck(PatternCheck):
    @classmethod
    def check_pattern(cls, pattern):
        """
        Checks that all nodes of the pattern use channels_last memory format.
        """
        for node in pattern:
            tensor = get_first_fake_tensor(node)
            if not is_channels_last(tensor):
                return False

        return True

    @classmethod
    def check_quantization_config(cls, quantization_config):
        """
        Checks that the quantization config uses per-tensor int8 quantization.
        """
        is_int8 = (
            quantization_config.input_activation.dtype == torch.int8
            and quantization_config.output_activation.dtype == torch.int8
        )
        return is_int8


class CortexMLinearCheck(PatternCheck):
    @classmethod
    def check_quantization_config(cls, quantization_config):
        """
        Checks that the quantization config uses per-tensor int8 quantization.
        """
        is_int8 = (
            quantization_config.input_activation.dtype == torch.int8
            and quantization_config.output_activation.dtype == torch.int8
        )
        return is_int8


class CortexMSoftmaxCheck(PatternCheck):

    @classmethod
    def check_pattern(cls, pattern):
        """
        Checks that given the tensor must either
        - be contiguous (default layout) with softmax dim == last logical dim, or
        - be channels_last with softmax dim == channel dim.
        """
        assert len(pattern) == 1
        node = pattern[0]

        tensor = get_first_fake_tensor(node)
        rank = len(tensor.shape)
        dim = node.args[1] % rank if len(node.args) > 1 else -1 % rank

        is_nhwc = is_channels_last(tensor)
        if is_nhwc:
            channel_dim = 1 if rank >= 2 else rank - 1
            if dim != channel_dim:
                return False
        else:
            if dim != rank - 1:
                return False

        return True

    @classmethod
    def check_quantization_config(cls, quantization_config):
        """
        Checks that the quantization config uses a valid configuration for CMSIS-NN softmax.
        """
        is_int8 = (
            quantization_config.input_activation.dtype == torch.int8
            and quantization_config.output_activation.dtype == torch.int8
        )
        is_per_tensor = cls.is_per_tensor(
            quantization_config.input_activation
        ) and cls.is_per_tensor(quantization_config.output_activation)
        correct_output_scale = (
            quantization_config.output_activation.scale == CMSIS_SOFTMAX_SCALE
        )
        correct_output_zero_point = (
            quantization_config.output_activation.zero_point == CMSIS_SOFTMAX_ZERO_POINT
        )

        return (
            is_int8
            and is_per_tensor
            and correct_output_scale
            and correct_output_zero_point
        )


class CortexMConvTranspose2DCheck(PatternCheck):

    @classmethod
    def _check_node(cls, node: Node) -> bool:
        if node is None:
            return False  # Reject if node is None

        tensor = get_first_fake_tensor(node)
        if tensor is None:
            return False  # Reject if no tensor found

        # REJECT if using NCHW format (we need channels_last/NHWC)
        if not is_channels_last(tensor):
            return False  # Reject NCHW

        # For aten.conv_transpose2d.input:
        #   (input, weight, bias, stride, padding, output_padding, groups, dilation)
        # Args: 5 = output_padding, 6 = groups, 7 = dilation
        if len(node.args) >= 6:
            output_padding = node.args[5]
            if isinstance(output_padding, (list, tuple)):
                if any(p != 0 for p in output_padding):
                    return False

        if len(node.args) >= 7:
            groups = node.args[6]
            if isinstance(groups, int) and groups > 1:
                return False

        if len(node.args) >= 8:
            dilation = node.args[7]
            if isinstance(dilation, (list, tuple)):
                if any(d != 1 for d in dilation):
                    return False
        return True

    @classmethod
    def check_pattern(cls, pattern):
        """
        Positive filter function for transpose conv to REJECT:
        1. NCHW memory format (we only support channels_last/NHWC)
        2. Grouped convolutions (groups > 1) - not supported by CMSIS-NN
        3. Non-zero output_padding - not supported by CMSIS-NN
        4. Dilation != 1 - produces incorrect results with CMSIS-NN

        Returns True to ACCEPT the node, False to REJECT.
        """
        for node in pattern:
            if not cls._check_node(node):
                return False  # REJECT invalid transpose conv

        return True  # ACCEPT channels_last transpose conv

    @classmethod
    def check_quantization_config(cls, quantization_config):
        """
        Checks that the quantization config uses per-tensor int8 quantization.
        """
        is_int8 = (
            quantization_config.input_activation.dtype == torch.int8
            and quantization_config.output_activation.dtype == torch.int8
        )
        is_ch_axis_1 = quantization_config.weight.ch_axis == 1

        return is_int8 and is_ch_axis_1
