# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from enum import IntEnum, unique
from functools import partial
from typing import Callable, Optional, Sequence, Set

import torch
from executorch.backends.qualcomm._passes.decompose_einsum import DecomposeEinsum
from executorch.backends.qualcomm._passes.decompose_silu import DecomposeSilu
from executorch.backends.qualcomm._passes.recompose_pixel_unshuffle import (
    RecomposePixelUnshuffle,
)
from executorch.backends.qualcomm._passes.reduce_dynamic_range import ReduceDynamicRange
from executorch.backends.qualcomm._passes.replace_inf_buffer import ReplaceInfBuffer
from executorch.backends.transforms.decompose_sdpa import (
    DecomposeScaledDotProductAttention,
)

from torch._ops import OpOverload
from torch.ao.quantization.quantizer import Quantizer
from torch.fx import GraphModule

from .annotators import OP_ANNOTATOR

from .qconfig import (
    get_16a16w_qnn_ptq_config,
    get_16a4w_qnn_ptq_config,
    get_16a4w_qnn_qat_config,
    get_16a8w_qnn_ptq_config,
    get_8a8w_qnn_ptq_config,
    get_8a8w_qnn_qat_config,
    get_ptq_per_channel_quant_config,
    get_qat_per_channel_quant_config,
    QuantizationConfig,
)

# To bypass the meta internal test error
get_default_16bit_qnn_ptq_config = get_16a16w_qnn_ptq_config

__all__ = [
    "QnnQuantizer",
    "QuantDtype",
    "get_16a4w_qnn_ptq_config",
    "get_16a8w_qnn_ptq_config",
    "get_16a16w_qnn_ptq_config",
    "get_8a8w_qnn_ptq_config",
    "get_8a8w_qnn_qat_config",
    "get_16a4w_qnn_qat_config",
]


@unique
class QuantDtype(IntEnum):
    """
    bits of activation and bits of weight
    """

    use_16a16w = 0
    use_16a8w = 1
    use_16a4w = 2
    use_8a8w = 3


quant_config_dict = {
    # PTQ
    (QuantDtype.use_16a16w, False): (
        get_16a16w_qnn_ptq_config,
        partial(
            get_ptq_per_channel_quant_config,
            act_dtype=torch.uint16,
            weight_dtype=torch.int16,
        ),
    ),
    (QuantDtype.use_16a8w, False): (
        get_16a8w_qnn_ptq_config,
        partial(
            get_ptq_per_channel_quant_config,
            act_dtype=torch.uint16,
            weight_dtype=torch.int8,
        ),
    ),
    (QuantDtype.use_16a4w, False): (
        get_16a4w_qnn_ptq_config,
        partial(
            get_ptq_per_channel_quant_config,
            act_dtype=torch.uint16,
            weight_dtype="int4",
        ),
    ),
    (QuantDtype.use_8a8w, False): (
        get_8a8w_qnn_ptq_config,
        partial(get_ptq_per_channel_quant_config),
    ),
    # QAT,
    (QuantDtype.use_16a4w, True): (
        get_16a4w_qnn_qat_config,
        partial(
            get_qat_per_channel_quant_config,
            act_dtype=torch.uint16,
            weight_dtype="int4",
        ),
    ),
    (QuantDtype.use_8a8w, True): (
        get_8a8w_qnn_qat_config,
        partial(get_qat_per_channel_quant_config),
    ),
}


class QnnQuantizer(Quantizer):
    SUPPORTED_OPS: Set = set(OP_ANNOTATOR.keys())

    def __init__(self):
        super().__init__()
        self.quant_ops: Set[OpOverload] = self.SUPPORTED_OPS.copy()

        self.is_qat = False
        self.quant_dtype = QuantDtype.use_8a8w
        self.quant_config: QuantizationConfig = get_8a8w_qnn_ptq_config()
        self.per_channel_quant_config = get_ptq_per_channel_quant_config()
        self.use_per_channel_weight_quant_ops: Set[OpOverload] = set()

        self.custom_quant_annotations: Sequence[Callable] = []
        self.discard_nodes: Set[str] = set()

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
            2. quant config
        """
        if isinstance(op, str):
            return

        if op in self.use_per_channel_weight_quant_ops:
            return self.per_channel_quant_config

        if op in self.quant_ops:
            return self.quant_config

        print(f"No quant config is implemented for op, {op}")

    def _update_per_channel_weight_quant_ops(self, ops: Set[OpOverload], enable: bool):
        if enable:
            self.use_per_channel_weight_quant_ops.update(ops)
        else:
            self.use_per_channel_weight_quant_ops.difference_update(ops)

    def add_custom_quant_annotations(
        self, custom_quant_annotations: Sequence[Callable]
    ) -> None:
        self.custom_quant_annotations = custom_quant_annotations

    def add_discard_nodes(self, nodes: Sequence[str]) -> None:
        self.discard_nodes = set(nodes)

    def add_discard_ops(self, ops: Sequence[OpOverload]) -> None:
        for op in ops:
            self.quant_ops.remove(op)

    def annotate(self, model: GraphModule) -> GraphModule:
        self._annotate(model)
        self._annotate_custom_annotation(model)

        return model

    def get_supported_ops(self) -> Set[OpOverload]:
        return self.SUPPORTED_OPS

    def set_quant_config(
        self, quant_dtype: QuantDtype, is_qat=False, act_observer=None
    ) -> None:
        self.quant_dtype = quant_dtype
        self.is_qat = is_qat
        if (quant_dtype, is_qat) not in quant_config_dict:
            raise RuntimeError(
                f"the quant config, (quant_dtype: {quant_dtype}, is_qat: {is_qat}) is not support"
            )

        quant_config_fuc, per_channel_quant_config_fuc = quant_config_dict[
            (quant_dtype, is_qat)
        ]
        self.quant_config = (
            quant_config_fuc(act_observer=act_observer)
            if act_observer
            else quant_config_fuc()
        )
        self.per_channel_quant_config = (
            per_channel_quant_config_fuc(act_observer=act_observer)
            if act_observer
            else per_channel_quant_config_fuc()
        )

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
        model = DecomposeEinsum()(model).graph_module
        model = ReplaceInfBuffer()(model).graph_module
        return model

    def validate(self, model: GraphModule) -> None:
        pass
