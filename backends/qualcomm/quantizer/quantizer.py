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
    get_16a8w_qnn_qat_config,
    get_8a4w_qnn_ptq_config,
    get_8a8w_qnn_ptq_config,
    get_8a8w_qnn_qat_config,
    get_ptq_per_block_quant_config,
    get_ptq_per_channel_quant_config,
    get_qat_per_block_quant_config,
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
    "get_16a8w_qnn_qat_config",
    "get_16a16w_qnn_ptq_config",
    "get_8a8w_qnn_ptq_config",
    "get_8a8w_qnn_qat_config",
    "get_8a4w_qnn_ptq_config",
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
    use_8a4w = 5


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
            weight_dtype=torch.int4,
        ),
        None,
    ),
    (QuantDtype.use_16a4w_block, False): (
        get_16a4w_qnn_ptq_config,
        partial(
            get_ptq_per_channel_quant_config,
            act_dtype=torch.uint16,
            weight_dtype=torch.int4,
        ),
        partial(
            get_ptq_per_block_quant_config,
            act_dtype=torch.uint16,
            weight_dtype=torch.int4,
        ),
    ),
    (QuantDtype.use_8a8w, False): (
        get_8a8w_qnn_ptq_config,
        partial(get_ptq_per_channel_quant_config),
        None,
    ),
    (QuantDtype.use_8a4w, False): (
        get_8a4w_qnn_ptq_config,
        partial(
            get_ptq_per_channel_quant_config,
            act_dtype=torch.uint8,
            weight_dtype=torch.int4,
        ),
        None,
    ),
    # QAT,
    (QuantDtype.use_16a4w, True): (
        get_16a4w_qnn_qat_config,
        partial(
            get_qat_per_channel_quant_config,
            act_dtype=torch.uint16,
            weight_dtype=torch.int4,
        ),
        None,
    ),
    (QuantDtype.use_16a4w_block, True): (
        get_16a4w_qnn_qat_config,
        partial(
            get_qat_per_channel_quant_config,
            act_dtype=torch.uint16,
            weight_dtype=torch.int4,
        ),
        partial(
            get_qat_per_block_quant_config,
            act_dtype=torch.uint16,
            weight_dtype=torch.int4,
        ),
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

        # Assume per_channel_quant/per_block_quant only happen on axis_0 or axis_1, increase the range if there's a need
        potential_axis = 2

        self.per_channel_quant_config_list = []
        for i in range(potential_axis):
            self.per_channel_quant_config_list.append(
                (
                    per_channel_quant_config_func(
                        act_observer=self.act_observer, ch_axis=i
                    )
                    if self.act_observer
                    else per_channel_quant_config_func(ch_axis=i)
                )
            )

        # Key is the node target, and value is the axis to perform per channel quantization
        self.op_axis_dict = {
            torch.ops.aten.conv1d.default: 0,
            torch.ops.aten.conv2d.default: 0,
            torch.ops.aten.conv3d.default: 0,
            torch.ops.aten.conv_transpose2d.input: 1,
            torch.ops.aten.conv_transpose3d.input: 1,
            torch.ops.aten.linear.default: 0,
        }

        self.use_per_channel_weight_quant_ops = {}
        if self.is_conv_per_channel:
            conv_ops = [
                torch.ops.aten.conv1d.default,
                torch.ops.aten.conv2d.default,
                torch.ops.aten.conv3d.default,
                torch.ops.aten.conv_transpose2d.input,
                torch.ops.aten.conv_transpose3d.input,
            ]
            self.use_per_channel_weight_quant_ops.update(
                {k: self.op_axis_dict[k] for k in conv_ops if k in self.op_axis_dict}
            )
        if self.is_linear_per_channel:
            linear_ops = [torch.ops.aten.linear.default]
            self.use_per_channel_weight_quant_ops.update(
                {k: self.op_axis_dict[k] for k in linear_ops if k in self.op_axis_dict}
            )

        if per_block_quant_config_func:
            self.per_block_quant_config_list = []
            for i in range(potential_axis):
                self.per_block_quant_config_list.append(
                    (
                        per_block_quant_config_func(
                            act_observer=self.act_observer, ch_axis=i
                        )
                        if self.act_observer
                        else per_block_quant_config_func(ch_axis=i)
                    )
                )


class QnnQuantizer(Quantizer):
    """
    QnnQuantizer is a quantization annotator designed for QNN backends.
    It uses OP_ANNOTATOR, a dictionary mapping OpOverload to annotator functions,
    to determine how each node should be annotated for quantization.

    Example usage:
        quantizer = QnnQuantizer()
        quantizer.set_default_quant_config(
            quant_dtype=QuantDtype.use_8a8w,
            is_qat=False,
            is_conv_per_channel=True,
            is_linear_per_channel=True,
            act_observer=MovingAverageMinMaxObserver,
        )
        quantizer.set_block_size_map({"conv2d": (1, 128, 1, 1)})
        quantizer.set_submodule_qconfig_list([
            (get_submodule_type_predicate("Add"), ModuleQConfig(quant_dtype=QuantDtype.use_16a4w))
        ])
        quantizer.add_custom_quant_annotations(...)
        quantizer.add_discard_nodes([node.name to skip annotation])
        quantizer.add_discard_ops([node.target to skip annotation])
    """

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
        self.recipe = None

    def _annotate(self, gm: GraphModule) -> None:
        """
        Annotates the nodes of the provided GraphModule in-place based on user defined quant configs during prepare_pt2e.

        For each node in the graph, nodes without quant config or those explicitly listed in `self.discard_nodes` are not annotated.
        """
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
        """
        Retrieves the `ModuleQConfig` for a given node by matching the first applicable callable function in the `submodule_qconfig_list`.
        You can add submodule-specific quant config using the `set_submodule_qconfig_list` method.

        Args:
            node (torch.fx.Node): The node for which to retrieve the quant config.

        Returns:
            ModuleQConfig: The matched submodule config, or the default config if no match is found.
        """
        for func, qconfig in self.submodule_qconfig_list:
            if func(node):
                return qconfig
        return self.default_quant_config

    def _get_quant_config(self, node: torch.fx.Node) -> Optional[QuantizationConfig]:
        """
        Select the quant config for a node based on priority.

        Priority order:
            1. Per-block quant config if block_size is set for node.
            2. Submodule-specific config if predicate matches.
            3. Per-channel config if op is in per-channel set.
            4. Default quant config if op is supported.

        Args:
            node (torch.fx.Node): The node to get quant config for.

        """
        op = node.target
        if isinstance(op, str):
            return
        config = self._get_submodule_qconfig(node)
        if block_size := self.block_size_map.get(node.name):
            ch_axis = config.op_axis_dict.get(node.target, 0)
            assert (
                len(config.per_block_quant_config_list) > ch_axis
            ), f"Unsupported per block quantization axis: {ch_axis}, please increase the range of per_block_quant_config_list"
            config = config.per_block_quant_config_list[ch_axis]
            config.block_size = block_size
            return config

        if op in config.use_per_channel_weight_quant_ops:
            ch_axis = config.use_per_channel_weight_quant_ops[op]
            assert (
                len(config.per_channel_quant_config_list) > ch_axis
            ), f"Unsupported per channel quantization axis: {ch_axis}, please increase the range of per_channel_quant_config_list"
            return config.per_channel_quant_config_list[ch_axis]

        if op in self.quant_ops:
            return config.quant_config

        print(f"No quant config is implemented for op, {op}")

    def add_custom_quant_annotations(
        self, custom_quant_annotations: Sequence[Callable]
    ) -> None:
        """
        Add custom annotation functions to be applied during prepare_pt2e.

        Args:
            custom_quant_annotations (Sequence[Callable]): A sequence of functions that take a GraphModule and perform custom annotation.
        """
        self.custom_quant_annotations = custom_quant_annotations

    def add_discard_nodes(self, nodes: Sequence[str]) -> None:
        """
        Specifies node IDs to exclude from quantization.
        """
        self.discard_nodes = set(nodes)

    def add_discard_ops(self, ops: Sequence[OpOverload]) -> None:
        """
        Specifies OpOverloads to exclude from quantization.
        """
        for op in ops:
            self.quant_ops.remove(op)

    def annotate(self, model: GraphModule) -> GraphModule:
        """
        Annotates GraphModule during prepare_pt2e.

        If a recipe is provided, it will be used to annotate the model.
        Otherwise, fallback to the default annotation flow.

        Args:
            model (GraphModule): The FX GraphModule to annotate.

        Returns:
            GraphModule: The annotated model.
        """
        if self.recipe:
            self.recipe.annotate(model)
        else:
            self._annotate(model)
            self._annotate_custom_annotation(model)

        return model

    def get_supported_ops(self) -> Set[OpOverload]:
        """
        Returns the set of supported OpOverloads for quantization.

        Returns:
            Set[OpOverload]: Supported ops.
        """
        return self.SUPPORTED_OPS

    def set_default_quant_config(
        self,
        quant_dtype: QuantDtype,
        is_qat=False,
        is_conv_per_channel=False,
        is_linear_per_channel=False,
        act_observer=None,
    ) -> None:
        """
        Set the default quant config for quantizer.

        Args:
            quant_dtype (QuantDtype): Specifies the quantized data type. By default, 8-bit activations and weights (8a8w) are used.
            is_qat (bool, optional): Enables Quantization-Aware Training (QAT) mode. Defaults to Post-Training Quantization (PTQ) mode.
            is_conv_per_channel (bool, optional): Enables per-channel quantization for convolution operations.
            is_linear_per_channel (bool, optional): Enables per-channel quantization for linear (fully connected) operations.
            act_observer (Optional[UniformQuantizationObserverBase], optional): Custom observer for activation quantization. If not specified, the default observer is determined by `QUANT_CONFIG_DICT`.

        """
        self.default_quant_config = ModuleQConfig(
            quant_dtype,
            is_qat=is_qat,
            is_conv_per_channel=is_conv_per_channel,
            is_linear_per_channel=is_linear_per_channel,
            act_observer=act_observer,
        )

    def set_block_size_map(self, block_size_map: Dict[str, Tuple]) -> None:
        """
        Set the mapping from node names to block sizes for per-block quantization.

        Args:
            block_size_map (Dict[str, Tuple]): Mapping from node name to block size.
        """
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
        """
        Applies QNN-specific transformation before annotation during prepare_pt2e.

        Args:
            model (GraphModule): The FX GraphModule to transform.

        Returns:
            GraphModule: The transformed model.
        """
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
