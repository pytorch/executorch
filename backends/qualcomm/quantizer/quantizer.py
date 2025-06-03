# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from enum import IntEnum, unique
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
from executorch.backends.qualcomm._passes.qnn_pass_manager import QnnPassManager

from torch._ops import OpOverload
from torch.fx import GraphModule
from torchao.quantization.pt2e import UniformQuantizationObserverBase
from torchao.quantization.pt2e.quantizer import Quantizer

from .annotators import OP_ANNOTATOR

from .qconfig import (
    get_16a16w_qnn_ptq_config,
    get_16a4w_qnn_ptq_config,
    get_16a4w_qnn_qat_config,
    get_16a8w_qnn_ptq_config,
    get_8a8w_qnn_ptq_config,
    get_8a8w_qnn_qat_config,
    get_ptq_per_block_quant_config,
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
    "get_ptq_per_block_quant_config",
]


@unique
class QuantDtype(IntEnum):
    """
    bits of activation and bits of weight
    """

    use_16a16w = 0
    use_16a8w = 1
    use_16a4w = 2
    use_16a4w_block = 3
    use_8a8w = 4


QUANT_CONFIG_DICT = {
    # PTQ
    (QuantDtype.use_16a16w, False): (
        get_16a16w_qnn_ptq_config,
        partial(
            get_ptq_per_channel_quant_config,
            act_dtype=torch.uint16,
            weight_dtype=torch.int16,
        ),
        None,
    ),
    (QuantDtype.use_16a8w, False): (
        get_16a8w_qnn_ptq_config,
        partial(
            get_ptq_per_channel_quant_config,
            act_dtype=torch.uint16,
            weight_dtype=torch.int8,
        ),
        None,
    ),
    (QuantDtype.use_16a4w, False): (
        get_16a4w_qnn_ptq_config,
        partial(
            get_ptq_per_channel_quant_config,
            act_dtype=torch.uint16,
            weight_dtype="int4",
        ),
        None,
    ),
    (QuantDtype.use_16a4w_block, False): (
        get_16a4w_qnn_ptq_config,
        partial(
            get_ptq_per_channel_quant_config,
            act_dtype=torch.uint16,
            weight_dtype="int4",
        ),
        partial(
            get_ptq_per_block_quant_config,
            act_dtype=torch.uint16,
            weight_dtype="int4",
        ),
    ),
    (QuantDtype.use_8a8w, False): (
        get_8a8w_qnn_ptq_config,
        partial(get_ptq_per_channel_quant_config),
        None,
    ),
    # QAT,
    (QuantDtype.use_16a4w, True): (
        get_16a4w_qnn_qat_config,
        partial(
            get_qat_per_channel_quant_config,
            act_dtype=torch.uint16,
            weight_dtype="int4",
        ),
        None,
    ),
    (QuantDtype.use_8a8w, True): (
        get_8a8w_qnn_qat_config,
        partial(get_qat_per_channel_quant_config),
        None,
    ),
}


@dataclass
class ModuleQConfig:
    quant_dtype: QuantDtype = QuantDtype.use_8a8w
    is_qat: bool = False
    is_conv_per_channel: bool = False
    is_linear_per_channel: bool = False
    act_observer: Optional[UniformQuantizationObserverBase] = None

    def __post_init__(self):
        if (self.quant_dtype, self.is_qat) not in QUANT_CONFIG_DICT:
            raise RuntimeError(
                f"the quant config, (quant_dtype: {self.quant_dtype}, is_qat: {self.is_qat}) is not support"
            )
        (
            quant_config_func,
            per_channel_quant_config_func,
            per_block_quant_config_func,
        ) = QUANT_CONFIG_DICT[(self.quant_dtype, self.is_qat)]
        self.quant_config = (
            quant_config_func(act_observer=self.act_observer)
            if self.act_observer
            else quant_config_func()
        )
        self.per_channel_quant_config = (
            per_channel_quant_config_func(act_observer=self.act_observer)
            if self.act_observer
            else per_channel_quant_config_func()
        )
        self.use_per_channel_weight_quant_ops = set()
        if self.is_conv_per_channel:
            self.use_per_channel_weight_quant_ops.update(
                {
                    torch.ops.aten.conv1d.default,
                    torch.ops.aten.conv2d.default,
                    torch.ops.aten.conv_transpose2d.input,
                }
            )
        if self.is_linear_per_channel:
            self.use_per_channel_weight_quant_ops.update(
                {
                    torch.ops.aten.linear.default,
                }
            )
        if per_block_quant_config_func:
            self.per_block_quant_config = (
                per_block_quant_config_func(act_observer=self.act_observer)
                if self.act_observer
                else per_block_quant_config_func()
            )


class QnnQuantizer(Quantizer):
    SUPPORTED_OPS: Set = set(OP_ANNOTATOR.keys())

    def __init__(self):
        super().__init__()
        self.quant_ops: Set[OpOverload] = self.SUPPORTED_OPS.copy()

        self.default_quant_config = ModuleQConfig()
        self.submodule_qconfig_list: List[
            Tuple[Callable[[torch.fx.Node], bool], ModuleQConfig]
        ] = []
        self.block_size_map = {}

        self.custom_quant_annotations: Sequence[Callable] = []
        self.discard_nodes: Set[str] = set()

    def _annotate(self, gm: GraphModule) -> None:
        for node in gm.graph.nodes:
            if node.name in self.discard_nodes:
                continue

            quant_config = self._get_quant_config(node)
            if quant_config:
                OP_ANNOTATOR[node.target](node, quant_config)

    def _annotate_custom_annotation(self, gm: GraphModule) -> None:
        for annotation_func in self.custom_quant_annotations:
            annotation_func(gm)

    def _get_submodule_qconfig(self, node: torch.fx.Node):
        for func, qconfig in self.submodule_qconfig_list:
            if func(node):
                return qconfig
        return self.default_quant_config

    def _get_quant_config(self, node: torch.fx.Node) -> Optional[QuantizationConfig]:
        """
        How to pick:
            1. is one of per_block_quant_config
            2. Pick specific submodule config if given.
            3. Pick one if op belongs to use_per_channel_weight_quant_ops
            4. If not 3, pick normal quant config
        """
        op = node.target
        if isinstance(op, str):
            return

        if block_size := self.block_size_map.get(node.name):
            config = self.default_quant_config.per_block_quant_config
            config.block_size = block_size
            return config

        config = self._get_submodule_qconfig(node)

        if op in config.use_per_channel_weight_quant_ops:
            return config.per_channel_quant_config

        if op in self.quant_ops:
            return config.quant_config

        print(f"No quant config is implemented for op, {op}")

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

    def set_default_quant_config(
        self,
        quant_dtype: QuantDtype,
        is_qat=False,
        is_conv_per_channel=False,
        is_linear_per_channel=False,
        act_observer=None,
    ) -> None:
        self.default_quant_config = ModuleQConfig(
            quant_dtype,
            is_qat,
            is_conv_per_channel,
            is_linear_per_channel,
            act_observer,
        )

    def set_block_size_map(self, block_size_map: Dict[str, Tuple]) -> None:
        self.block_size_map = block_size_map

    def set_submodule_qconfig_list(
        self, submodule_qconfig_list: List[Tuple[Callable, ModuleQConfig]]
    ) -> None:
        """
        Set specific quant config from a callback function.
        If a node fits more than one callback, only apply the first one.
        """
        self.submodule_qconfig_list = submodule_qconfig_list

    def transform_for_annotation(self, model: GraphModule) -> GraphModule:
        return QnnPassManager().transform_for_annotation_pipeline(model)

    def validate(self, model: GraphModule) -> None:
        pass


def get_submodule_type_predicate(module_type_str):
    """
    An example of nn_module_stack
    {
        'L__self__': ('', 'executorch.backends.qualcomm.tests.models.SubModules'),
        'L__self___add': ('add', 'executorch.backends.qualcomm.tests.models.Add')
    }
    """

    def predicate(node):
        if nn_module_stack := node.meta.get("nn_module_stack"):
            for _, type_name in nn_module_stack.values():
                if module_type_str in type_name:
                    return True
        return False

    return predicate


def get_submodule_name_predicate(module_name_str):
    """
    An example of nn_module_stack
    {
        'L__self__': ('', 'executorch.backends.qualcomm.tests.models.SubModules'),
        'L__self___add': ('add', 'executorch.backends.qualcomm.tests.models.Add')
    }
    """

    def predicate(node):
        if nn_module_stack := node.meta.get("nn_module_stack"):
            for name in nn_module_stack.keys():
                if module_name_str in name:
                    return True
        return False

    return predicate
