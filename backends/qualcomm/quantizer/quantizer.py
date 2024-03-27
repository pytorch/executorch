# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from enum import IntEnum, unique
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
from executorch.backends.qualcomm.passes.convert_hardsigmoid import ConvertHardsigmoid
from executorch.backends.qualcomm.passes.decompose_scaled_dot_product_attention import (
    DecomposeScaledDotProductAttention,
)
from executorch.backends.qualcomm.passes.decompose_silu import DecomposeSilu
from executorch.backends.qualcomm.passes.reduce_dynamic_range import ReduceDynamicRange
from executorch.backends.qualcomm.passes.remove_clone import RemoveClone
from executorch.backends.qualcomm.passes.replace_inf_buffer import ReplaceInfBuffer

from torch import Tensor
from torch._ops import OpOverload
from torch.ao.quantization.observer import (
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    PerChannelMinMaxObserver,
)
from torch.ao.quantization.quantizer import (
    DerivedQuantizationSpec,
    QuantizationSpec,
    Quantizer,
)

from torch.fx import GraphModule, Node

from .utils import OP_ANNOTATOR, QuantizationConfig

__all__ = [
    "QnnQuantizer",
    "QuantDtype",
    "get_16a4w_qnn_ptq_config",
    "get_default_16bit_qnn_ptq_config",
    "get_default_8bit_qnn_ptq_config",
]


@unique
class QuantDtype(IntEnum):
    """
    bits of activation and bits of weight
    """

    use_16a16w = 0
    use_16a4w = 1
    use_8a8w = 2


def _derived_bias_quant_spec(node: Node) -> DerivedQuantizationSpec:
    def _derive_bias_qparams_fn(
        obs_or_fqs: List,
    ) -> Tuple[Tensor, Tensor]:
        assert (
            len(obs_or_fqs) == 2
        ), f"Expecting two obs/fqs, one for activation and one for weight, got: {len(obs_or_fqs)}"
        act_obs_or_fq = obs_or_fqs[0]
        weight_obs_or_fq = obs_or_fqs[1]
        weight_scale, weight_zp = weight_obs_or_fq.calculate_qparams()
        act_scale, act_zp = act_obs_or_fq.calculate_qparams()
        (broadcast_act_scale, broadcast_weight_scale) = torch.broadcast_tensors(
            act_scale, weight_scale
        )
        derived_scale = (broadcast_act_scale * broadcast_weight_scale).to(torch.float32)
        derived_zero = torch.zeros(derived_scale.size()).to(torch.int32)
        return (derived_scale, derived_zero)

    input_act = node.args[0]
    assert isinstance(input_act, Node)
    weight = node.args[1]
    assert isinstance(weight, Node)

    return DerivedQuantizationSpec(
        derived_from=[(input_act, node), (weight, node)],
        derive_qparams_fn=_derive_bias_qparams_fn,
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.int32).min,
        quant_max=torch.iinfo(torch.int32).max,
        ch_axis=0,
        qscheme=torch.per_channel_symmetric,
    )


def get_default_8bit_qnn_ptq_config() -> QuantizationConfig:
    extra_args: Dict[str, Any] = {"eps": 2**-12}

    act_quantization_spec = QuantizationSpec(
        dtype=torch.uint8,
        quant_min=0,
        quant_max=torch.iinfo(torch.uint8).max,
        qscheme=torch.per_tensor_affine,
        observer_or_fake_quant_ctr=MovingAverageMinMaxObserver.with_args(**extra_args),
    )

    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=torch.iinfo(torch.int8).min + 1,
        quant_max=torch.iinfo(torch.int8).max,
        qscheme=torch.per_tensor_symmetric,
        ch_axis=0,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(**extra_args),
    )

    bias_quantization_spec = QuantizationSpec(
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.int32).min,
        quant_max=torch.iinfo(torch.int32).max,
        qscheme=torch.per_tensor_symmetric,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(**extra_args),
    )

    quantization_config = QuantizationConfig(
        input_activation=act_quantization_spec,
        output_activation=act_quantization_spec,
        weight=weight_quantization_spec,
        bias=bias_quantization_spec,
    )

    return quantization_config


# 4 bits quantization only supports specific ops.
def get_16a4w_qnn_ptq_config() -> QuantizationConfig:
    extra_args: Dict[str, Any] = {"eps": 2**-20}
    act_quantization_spec = QuantizationSpec(
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.uint16).min,
        quant_max=torch.iinfo(torch.uint16).max,
        qscheme=torch.per_tensor_affine,
        observer_or_fake_quant_ctr=MovingAverageMinMaxObserver.with_args(**extra_args),
    )

    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-7,
        quant_max=7,
        qscheme=torch.per_tensor_symmetric,
        ch_axis=0,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(**extra_args),
    )

    bias_quantization_spec = QuantizationSpec(
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.int32).min,
        quant_max=torch.iinfo(torch.int32).max,
        qscheme=torch.per_tensor_symmetric,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(**extra_args),
    )

    quantization_config = QuantizationConfig(
        input_activation=act_quantization_spec,
        output_activation=act_quantization_spec,
        weight=weight_quantization_spec,
        bias=bias_quantization_spec,
    )

    return quantization_config


def get_default_16bit_qnn_ptq_config() -> QuantizationConfig:
    extra_args: Dict[str, Any] = {"eps": 2**-20}
    act_quantization_spec = QuantizationSpec(
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.uint16).min,
        quant_max=torch.iinfo(torch.uint16).max,
        qscheme=torch.per_tensor_affine,
        observer_or_fake_quant_ctr=MovingAverageMinMaxObserver.with_args(**extra_args),
    )

    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int16,
        quant_min=torch.iinfo(torch.int16).min + 1,
        quant_max=torch.iinfo(torch.int16).max,
        qscheme=torch.per_tensor_symmetric,
        ch_axis=0,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(**extra_args),
    )

    # torch does not support uint16 quantization, use int32 to bypass
    bias_quantization_spec = QuantizationSpec(
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.int32).min,
        quant_max=torch.iinfo(torch.int32).max,
        qscheme=torch.per_tensor_symmetric,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(**extra_args),
    )

    quantization_config = QuantizationConfig(
        input_activation=act_quantization_spec,
        output_activation=act_quantization_spec,
        weight=weight_quantization_spec,
        bias=bias_quantization_spec,
    )

    return quantization_config


def get_ptq_per_channel_weight_config(
    act_dtype=torch.uint8, weight_dtype=torch.int8
) -> QuantizationConfig:
    extra_args: Dict[str, Any] = {"eps": 2**-12}

    supported_act_types = {
        torch.uint8,
        torch.uint16,
        torch.int8,
        torch.int16,
    }
    # TODO accept "int4" temporally. Remove "int4" when torch support torch.int4 dtype
    supported_weight_dtypes = {"int4", torch.int8, torch.int16}
    assert (
        act_dtype in supported_act_types
    ), f"act_dtype, {act_dtype} is not one of supported types, {supported_act_types}"

    assert (
        weight_dtype in supported_weight_dtypes
    ), f"weight_dtype, {weight_dtype} is not one of supported types, {supported_weight_dtypes}"

    # torch do not support uint16 quantization, use int32 to bypass
    act_quantization_spec = QuantizationSpec(
        dtype=torch.int32 if act_dtype == torch.uint16 else act_dtype,
        quant_min=torch.iinfo(act_dtype).min,
        quant_max=torch.iinfo(act_dtype).max,
        qscheme=torch.per_tensor_affine,
        observer_or_fake_quant_ctr=HistogramObserver.with_args(**extra_args),
    )

    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8 if weight_dtype == "int4" else weight_dtype,
        quant_min=-7 if weight_dtype == "int4" else torch.iinfo(weight_dtype).min + 1,
        quant_max=7 if weight_dtype == "int4" else torch.iinfo(weight_dtype).max,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
        observer_or_fake_quant_ctr=PerChannelMinMaxObserver.with_args(**extra_args),
    )

    bias_quantization_spec = _derived_bias_quant_spec

    quantization_config = QuantizationConfig(
        input_activation=act_quantization_spec,
        output_activation=act_quantization_spec,
        weight=weight_quantization_spec,
        bias=bias_quantization_spec,
    )

    return quantization_config


class QnnQuantizer(Quantizer):
    SUPPORTED_OPS: Set = set(OP_ANNOTATOR.keys())

    def __init__(self):
        super().__init__()
        self.bit8_quant_config: QuantizationConfig = get_default_8bit_qnn_ptq_config()
        self.bit16_quant_config: QuantizationConfig = get_default_16bit_qnn_ptq_config()

        self.bit8_quant_ops: Set[OpOverload] = self.SUPPORTED_OPS.copy()
        self.bit16_quant_ops: Set[OpOverload] = set()

        self.custom_quant_annotations: Sequence[Callable] = []
        self.discard_nodes: Set[str] = set()

        self.use_per_channel_weight_quant_ops: Set[OpOverload] = set()
        # the weight quantized for activation 8 bits and 16 bits
        self.per_channel_weight_dtype: Dict = {
            "8bit_act": torch.int8,
            "16bit_act": torch.int16,
        }

    def _annotate(self, gm: GraphModule) -> None:
        for node in gm.graph.nodes:
            if node.name in self.discard_nodes:
                continue

            quant_config = self._get_quant_config(node.target)
            if quant_config:
                OP_ANNOTATOR[node.target](node, quant_config)

    def _annotate_custom_annotation(self, gm: GraphModule) -> None:
        for annotation_func in self.custom_quant_annotations:
            annotation_func(gm)

    def _get_quant_config(self, op: str | OpOverload) -> Optional[QuantizationConfig]:
        """
        Priority:
            1. is one of use_per_channel_weight_quant_ops
            2. int8 / int16 config
        """
        if type(op) == str:
            return

        if op in self.use_per_channel_weight_quant_ops:
            if op in self.bit16_quant_ops:
                return get_ptq_per_channel_weight_config(
                    torch.uint16, self.per_channel_weight_dtype["16bit_act"]
                )
            return get_ptq_per_channel_weight_config(
                weight_dtype=self.per_channel_weight_dtype["8bit_act"]
            )

        if op in self.bit8_quant_ops:
            return self.bit8_quant_config

        if op in self.bit16_quant_ops:
            return self.bit16_quant_config

        print(f"No quant config is implemented for op, {op}")

    def _update_per_channel_weight_quant_ops(self, ops: Set[OpOverload], enable: bool):
        if enable:
            self.use_per_channel_weight_quant_ops.update(ops)
        else:
            self.use_per_channel_weight_quant_ops.difference(ops)

    def add_16bit_quant_ops(self, ops: Set[OpOverload]) -> None:
        for op in ops:
            assert (
                op in self.SUPPORTED_OPS
            ), f"The annotation of op {op} is not implemented"

            self.bit8_quant_ops.remove(op)
            self.bit16_quant_ops.add(op)

    def add_custom_quant_annotations(
        self, custom_quant_annotations: Sequence[Callable]
    ) -> None:
        self.custom_quant_annotations = custom_quant_annotations

    def add_discard_nodes(self, nodes: Sequence[str]) -> None:
        self.discard_nodes = set(nodes)

    def add_discard_ops(self, ops: Sequence[OpOverload]) -> None:
        for op in ops:
            if op in self.bit8_quant_ops:
                self.bit8_quant_ops.remove(op)
            if op in self.bit16_quant_ops:
                self.bit16_quant_ops.remove(op)

    def annotate(self, model: GraphModule) -> GraphModule:
        self._annotate(model)
        self._annotate_custom_annotation(model)

        return model

    def get_supported_ops(self) -> Set[OpOverload]:
        return self.SUPPORTED_OPS

    def set_bit16_op_quant_config(
        self, quantization_config: QuantizationConfig
    ) -> None:
        self.bit16_quant_config = quantization_config

    def set_bit8_op_quant_config(self, quantization_config: QuantizationConfig) -> None:
        self.bit8_quant_config = quantization_config

    def set_per_channel_weight_dtype(
        self,
        weight_dtype_for_8bit_act: Optional[str | torch.dtype] = None,
        weight_dtype_for_16bit_act: Optional[str | torch.dtype] = None,
    ) -> None:
        # TODO accept temporally str type. Remove it when torch support torch.int4 dtype
        if weight_dtype_for_8bit_act:
            self.per_channel_weight_dtype["8bit_act"] = weight_dtype_for_8bit_act
        if weight_dtype_for_16bit_act:
            self.per_channel_weight_dtype["16bit_act"] = weight_dtype_for_16bit_act

    def set_per_channel_conv_quant(self, enable: bool) -> None:
        conv_ops = {torch.ops.aten.conv1d.default, torch.ops.aten.conv2d.default}
        self._update_per_channel_weight_quant_ops(conv_ops, enable)

    def set_per_channel_linear_quant(self, enable: bool) -> None:
        linear_ops = {
            torch.ops.aten.linear.default,
        }
        self._update_per_channel_weight_quant_ops(linear_ops, enable)

    def transform_for_annotation(self, model: GraphModule) -> GraphModule:
        model = RemoveClone()(model).graph_module
        model = ReduceDynamicRange()(model).graph_module
        model = ConvertHardsigmoid(quantization_capture=True)(model).graph_module
        model = DecomposeScaledDotProductAttention()(model).graph_module
        model = DecomposeSilu()(model).graph_module
        model = ReplaceInfBuffer()(model).graph_module

        return model

    def validate(self, model: GraphModule) -> None:
        pass
