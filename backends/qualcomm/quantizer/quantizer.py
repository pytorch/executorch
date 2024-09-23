# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from enum import IntEnum, unique
from typing import Callable, Dict, Optional, Sequence, Set

import torch
from executorch.backends.qualcomm.passes.decompose_silu import DecomposeSilu
from executorch.backends.qualcomm.passes.recompose_pixel_unshuffle import (
    RecomposePixelUnshuffle,
)
from executorch.backends.qualcomm.passes.reduce_dynamic_range import ReduceDynamicRange
from executorch.backends.qualcomm.passes.replace_inf_buffer import ReplaceInfBuffer
from executorch.backends.transforms.decompose_sdpa import (
    DecomposeScaledDotProductAttention,
)

from torch._ops import OpOverload
from torch.ao.quantization.quantizer import Quantizer
from torch.fx import GraphModule

from .utils import (
    get_16a4w_qnn_ptq_config,
    get_16a8w_qnn_ptq_config,
    get_default_16bit_qnn_ptq_config,
    get_default_8bit_qnn_ptq_config,
    get_ptq_per_channel_weight_config,
    OP_ANNOTATOR,
    QuantizationConfig,
)

__all__ = [
    "QnnQuantizer",
    "QuantDtype",
    "get_16a4w_qnn_ptq_config",
    "get_16a8w_qnn_ptq_config",
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
        if isinstance(op, str):
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
            self.use_per_channel_weight_quant_ops.difference_update(ops)

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
        model = ReduceDynamicRange()(model).graph_module
        model = RecomposePixelUnshuffle(quantization_capture=True)(model).graph_module
        model = DecomposeScaledDotProductAttention()(model).graph_module
        model = DecomposeSilu()(model).graph_module
        model = ReplaceInfBuffer()(model).graph_module
        return model

    def validate(self, model: GraphModule) -> None:
        pass
