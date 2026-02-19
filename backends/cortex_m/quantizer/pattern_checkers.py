# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

import torch
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.cortex_m.passes.passes_utils import (
    coerce_int_pair,
    is_channel_broadcast,
    is_channels_last,
)
from executorch.backends.cortex_m.quantizer.quantization_configs import (
    CMSIS_SOFTMAX_SCALE,
    CMSIS_SOFTMAX_ZERO_POINT,
    CortexMQuantizationConfig,
)
from torch.fx import Node
from torchao.quantization.pt2e.quantizer import (
    QuantizationSpecBase,
    SharedQuantizationSpec,
)


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
    def is_per_tensor(cls, qspec: QuantizationSpecBase | None) -> bool:
        """
        Returns true if the given quantization spec is per-tensor, otherwise false.
        """
        if not isinstance(qspec, QuantizationSpecBase):
            return False
        return qspec.qscheme in (torch.per_tensor_affine, torch.per_tensor_symmetric)

    @classmethod
    def is_per_channel(cls, qspec: QuantizationSpecBase | None) -> bool:
        """
        Returns true if the given quantization spec is per-channel, otherwise false.
        """
        if not isinstance(qspec, QuantizationSpecBase):
            return False
        return qspec.qscheme in (torch.per_channel_affine, torch.per_channel_symmetric)

    @classmethod
    def is_int8_activations(
        cls, qconfig: CortexMQuantizationConfig, output_node: Node | None = None
    ) -> bool:
        """
        Returns true if the given quantization spec uses int8 quantization, otherwise false.

        Output node is required for determining output quantization spec for some ops, otherwise it can be left as None.
        """
        input_qspec = qconfig.get_input_act_qspec()
        output_qspec = qconfig.get_output_act_qspec(output_node)
        if not isinstance(input_qspec, QuantizationSpecBase) or not isinstance(
            output_qspec, QuantizationSpecBase
        ):
            return False
        return input_qspec.dtype == torch.int8 and output_qspec.dtype == torch.int8

    @classmethod
    def check_pattern(cls, pattern: list[Node]) -> bool:
        """
        Returns true if the given pattern is supported, otherwise false.
        """
        return True

    @classmethod
    def check_quantization_config(
        cls, pattern: list[Node], quantization_config: CortexMQuantizationConfig
    ) -> bool:
        """
        Returns true if the given quantization config is supported for a given node pattern, otherwise false.
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
    def check_quantization_config(
        cls, pattern: list[Node], quantization_config: CortexMQuantizationConfig
    ):
        """
        Checks that the quantization config uses per-tensor int8 quantization.
        """
        is_per_tensor = PatternCheck.is_per_tensor(
            quantization_config.get_input_act_qspec()
        ) and PatternCheck.is_per_tensor(quantization_config.get_output_act_qspec())
        is_int8 = cls.is_int8_activations(quantization_config)
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
    def check_quantization_config(
        cls, pattern: list[Node], quantization_config: CortexMQuantizationConfig
    ):
        """
        Checks that the quantization config uses per-tensor int8 quantization.
        """
        is_int8 = cls.is_int8_activations(quantization_config)
        conv_node = pattern[0] if pattern else None
        weight_qspec = quantization_config.get_weight_qspec(conv_node)
        is_ch_axis_0 = (
            weight_qspec.ch_axis == 0 or weight_qspec.ch_axis is None
        )  # Accept if ch_axis is 0 or not specified (default to per-tensor)
        return is_int8 and is_ch_axis_0


class CortexMLinearCheck(PatternCheck):
    @classmethod
    def check_quantization_config(
        cls, pattern: list[Node], quantization_config: CortexMQuantizationConfig
    ):
        """
        Checks that the quantization config uses per-tensor int8 quantization.
        """
        is_int8 = cls.is_int8_activations(quantization_config)
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
    def check_quantization_config(
        cls, pattern: list[Node], quantization_config: CortexMQuantizationConfig
    ):
        """
        Checks that the quantization config uses a valid configuration for CMSIS-NN softmax.
        """
        output_node = pattern[-1] if pattern else None
        input_qspec = quantization_config.get_input_act_qspec()
        output_qspec = quantization_config.get_output_act_qspec(output_node)

        is_int8 = cls.is_int8_activations(quantization_config, output_node)
        is_per_tensor = cls.is_per_tensor(input_qspec) and cls.is_per_tensor(
            output_qspec
        )
        correct_output_scale = output_qspec.scale == CMSIS_SOFTMAX_SCALE
        correct_output_zero_point = output_qspec.zero_point == CMSIS_SOFTMAX_ZERO_POINT

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
    def check_quantization_config(
        cls, pattern: list[Node], quantization_config: CortexMQuantizationConfig
    ):
        """
        Checks that the quantization config uses per-tensor int8 quantization.
        """
        is_int8 = cls.is_int8_activations(quantization_config)

        transpose_conv_node = pattern[0] if pattern else None
        weight_qspec = quantization_config.get_weight_qspec(transpose_conv_node)
        is_ch_axis_1 = (
            weight_qspec.ch_axis == 1 or weight_qspec.ch_axis is None
        )  # Accept if ch_axis is 1 or not specified (default to per-tensor)

        return is_int8 and is_ch_axis_1


class CortexMAvgPool2DCheck(PatternCheck):
    @classmethod
    def check_pattern(cls, pattern):
        if not pattern:
            return False
        node = pattern[0]
        ceil_mode = cast(bool, node.args[4]) if len(node.args) > 4 else False
        count_include_pad = cast(bool, node.args[5]) if len(node.args) > 5 else True
        return not (ceil_mode or count_include_pad)

    @classmethod
    def check_quantization_config(
        cls, pattern: list[Node], quantization_config: CortexMQuantizationConfig
    ):
        output_node = pattern[-1] if pattern else None
        input_qspec = quantization_config.get_input_act_qspec()
        output_qspec = quantization_config.get_output_act_qspec(output_node)
        if isinstance(output_qspec, SharedQuantizationSpec):
            output_qspec = input_qspec
        if input_qspec is None or output_qspec is None:
            return False
        is_int8 = input_qspec.dtype == torch.int8 and output_qspec.dtype == torch.int8
        is_per_tensor = cls.is_per_tensor(input_qspec) and cls.is_per_tensor(
            output_qspec
        )
        return is_int8 and is_per_tensor


class CortexMMaxPool2DCheck(PatternCheck):
    @classmethod
    def _pool_arg_as_bool(cls, node: Node, index: int, default: bool) -> bool:
        raw = node.args[index] if len(node.args) > index else default
        if raw is None:
            return default
        return bool(raw)

    @classmethod
    def check_pattern(cls, pattern):
        if not pattern:
            return False
        node = pattern[0]
        raw_dilation = node.args[4] if len(node.args) > 4 else (1, 1)
        dilation = coerce_int_pair(raw_dilation, (1, 1))
        ceil_mode = cls._pool_arg_as_bool(node, 5, False)
        if dilation != (1, 1) or ceil_mode:
            meta_custom = node.meta.get("custom", {})
            cortex_m_meta = meta_custom.get("cortex_m", {})
            cortex_m_meta["skip_quantized_max_pool2d"] = True
            meta_custom["cortex_m"] = cortex_m_meta
            node.meta["custom"] = meta_custom
        return True

    @classmethod
    def check_quantization_config(
        cls, pattern: list[Node], quantization_config: CortexMQuantizationConfig
    ):
        maxpool_node = pattern[0]
        input_qspec = quantization_config.get_input_act_qspec()
        output_qspec = quantization_config.get_output_act_qspec(maxpool_node)
        if not isinstance(output_qspec, SharedQuantizationSpec):
            return False
        is_int8 = input_qspec.dtype == torch.int8
        is_per_tensor = cls.is_per_tensor(input_qspec)
        return is_int8 and is_per_tensor
