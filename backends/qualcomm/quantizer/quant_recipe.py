# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import re
from abc import ABC, abstractmethod
from enum import IntEnum, unique
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
from executorch.backends.qualcomm.quantizer.quantizer import (
    ModuleQConfig,
    QnnQuantizer,
    QuantDtype,
    QuantizationConfig,
)
from tabulate import tabulate
from torch._ops import OpOverload
from torchao.quantization.pt2e import UniformQuantizationObserverBase

from .annotators import OP_ANNOTATOR


def extract_node_metadata_mapping(node: torch.fx.Node):
    deepest_module = None

    if node.op == "call_function" and "nn_module_stack" in node.meta:
        deepest_module = list(node.meta["nn_module_stack"].values())[-1][0]

    return deepest_module


@unique
class QuantGranularity(IntEnum):
    """
    Defines the quantization granularity levels:
    - PER_TENSOR: single scale offset for entire tensor.
    - PER_CHANNEL: independent scale/offset per channel within tensor.
    - PER_BLOCK:  independent scale/offset per block within tensor.
    """

    PER_TENSOR = 0
    PER_CHANNEL = 1
    PER_BLOCK = 2


class QuantizationStrategy(ABC):
    """
    Abstract base class for strategies that assign quantization config to FX graph nodes.

    Each strategy defines how to match nodes (e.g., by operator target, module stack pattern)
    and provides a corresponding quantization config when a match occurs.

    Attributes:
        quant_dtype (QuantDtype): Data type for quantization (e.g., 16a8w, 16a4w).
        is_qat (bool): Whether the strategy applies QAT (True) or PTQ (False).
        granularity (QuantGranularity): Quantization granularity (PER_TENSOR, PER_CHANNEL, PER_BLOCK).
        act_observer (UniformQuantizationObserverBase): Observer class for activation quantization.
        extra_kwargs (Dict): Additional configuration parameters (e.g., block size).
        note (str): Developer notes or comments.
        priority (int): Priority for resolving conflicts among multiple strategies.

    Abstract Methods:
        _matches(node): Return True if the node matches this strategy's criteria.
    """

    def __init__(
        self,
        quant_dtype: QuantDtype,
        is_qat: bool,
        granularity: QuantGranularity,
        act_observer: UniformQuantizationObserverBase,
        extra_kwargs: Dict,
        note: str,
        priority: int,
    ):
        self.quant_dtype = quant_dtype
        self.is_qat = is_qat
        self.granularity = granularity
        self.act_observer = act_observer
        self.extra_kwargs = extra_kwargs
        self.note = note
        self.priority = priority

        self.quant_config = ModuleQConfig(
            quant_dtype=self.quant_dtype,
            is_qat=self.is_qat,
            is_conv_per_channel=True,
            is_linear_per_channel=True,
            act_observer=self.act_observer,
        )

    @abstractmethod
    def _matches(self, node: torch.fx.Node) -> bool:
        pass

    def get_quant_config(self, node: torch.fx.Node) -> Optional[QuantizationConfig]:
        op: OpOverload = node.target

        if not self._matches(node):
            return None

        if self.granularity == QuantGranularity.PER_TENSOR:
            return self.quant_config.quant_config
        elif self.granularity == QuantGranularity.PER_CHANNEL:
            ch_axis = self.quant_config.use_per_channel_weight_quant_ops.get(op)
            assert (
                ch_axis is not None
                and len(self.quant_config.per_channel_quant_config_list) > ch_axis
            ), f"Unsupported per channel quantization axis: {ch_axis}, please increase the range of per_channel_quant_config_list"
            return self.quant_config.per_channel_quant_config_list[ch_axis]
        elif self.granularity == QuantGranularity.PER_BLOCK:
            ch_axis = self.quant_config.op_axis_dict.get(op)
            assert (
                ch_axis is not None
                and len(self.quant_config.per_block_quant_config_list) > ch_axis
            ), f"Unsupported per block quantization axis: {ch_axis}, please increase the range of per_block_quant_config_list"
            config = self.quant_config.per_block_quant_config_list[ch_axis]
            config.block_size = self.extra_kwargs["block_size"]
            return config
        else:
            raise ValueError(
                f"Unsupported quantization granularity: {self.granularity}. "
                f"Supported values: {[granularity.name for granularity in QuantGranularity]}"
            )


class ByNodeTarget(QuantizationStrategy):
    """
    Strategy that assigns quantization config to nodes based on their op target.
    Useful for applying quantization to specific operations such as `aten.conv2d` or `aten.linear`.

    Attributes:
        targets (Set[OpOverload]): Set of op overloads to match against node targets.
    """

    def __init__(
        self,
        quant_dtype,
        is_qat,
        granularity,
        act_observer,
        extra_kwargs,
        note,
        priority,
        targets: Set[OpOverload],
    ):
        super().__init__(
            quant_dtype,
            is_qat,
            granularity,
            act_observer,
            extra_kwargs,
            note,
            priority,
        )
        self.targets = targets

    def _matches(self, node: torch.fx.Node) -> bool:
        # Matching: A node matches if its `node.target` is in the `targets` set.
        return node.target in self.targets


class ByNameRegex(QuantizationStrategy):
    """
    Strategy that assigns quantization config to nodes whose module stack matches given regex patterns.
    Useful for targeting layers by name patterns (e.g., "layers.[0-3].feed_forward" or "layers.*.attention") in the module hierarchy.

    Attributes:
        patterns (Set[str]): Set of regex patterns to match against module stack paths.
    """

    def __init__(
        self,
        quant_dtype,
        is_qat,
        granularity,
        act_observer,
        extra_kwargs,
        note,
        priority,
        patterns: Set[str],
    ):
        super().__init__(
            quant_dtype,
            is_qat,
            granularity,
            act_observer,
            extra_kwargs,
            note,
            priority,
        )
        self.patterns = patterns

    def _matches(self, node: torch.fx.Node) -> bool:
        # Matching: A node matches if its `nn_module_stack` metadata contains a module path that matches any regex pattern.
        if node.op == "call_function" and "nn_module_stack" in node.meta:
            for module_stack, _ in list(node.meta["nn_module_stack"].values())[::-1]:
                if module_stack and any(
                    re.search(p, module_stack) for p in self.patterns
                ):
                    return True
        return False


class QuantRecipe:
    """
    A QuantRecipe builder for defining quantization strategies to an FX GraphModule.

    QuantRecipe manages a collection of strategies (e.g., by operator target or regex pattern)
    and applies them to nodes in an FX graph to produce fine-grained quantization annotations.

    Attributes:
        verbose (bool): If True, prints a summary after annotation.
        custom_quant_annotations (Sequence[Callable]): Custom annotation functions applied after strategies.

        _strategies (List[QuantizationStrategy]): Registered quantization strategies.
        _pending_annotate_nodes (Dict[torch.fx.Node, Tuple[QuantizationConfig, QuantizationStrategy]]):
            Internal mapping of nodes to their resolved quantization config and strategy.
    """

    def __init__(
        self,
        quant_dtype,
        is_qat,
        act_observer: UniformQuantizationObserverBase,
        granularity: QuantGranularity,
        note: str = "",
        extra_kwargs: Optional[dict] = None,
        verbose: bool = False,
    ):
        """
        Initialize a QuantRecipe with a default quantization strategy.

        Args:
            quant_dtype (QuantDtype): Data type for quantization (e.g., int8, int4).
            is_qat (bool): Whether to apply QAT (True) or PTQ (False).
            act_observer (UniformQuantizationObserverBase): Observer class for activation quantization.
            granularity (QuantGranularity): Quantization granularity (PER_TENSOR, PER_CHANNEL, PER_BLOCK).
            note (str): Optional description for the default strategy.
            extra_kwargs (dict, optional): Additional parameters (e.g., block size, group size).
            verbose (bool): If True, prints a summary table after annotation.
        """

        self.verbose = verbose
        self.custom_quant_annotations: Sequence[Callable] = []

        self._strategies: List[QuantizationStrategy] = []
        self._pending_annotate_nodes: Dict[
            torch.fx.Node, Tuple[QuantizationConfig, QuantizationStrategy]
        ] = {}
        self._default_strategy = ByNodeTarget(
            quant_dtype,
            is_qat,
            granularity,
            act_observer,
            extra_kwargs or {},
            note,
            priority=1,
            targets=QnnQuantizer.SUPPORTED_OPS,
        )

    def _annotate_custom_annotation(self, gm: torch.fx.GraphModule) -> None:
        for annotation_func in self.custom_quant_annotations:
            annotation_func(gm)

    def annotate(self, graph_module: torch.fx.GraphModule):
        # Sort node level strategies by (priority, insertion index).
        # Higher priority value comes first; if priorities are equal, original insertion order is preserved.
        strategies: List[QuantizationStrategy] = [
            strategy
            for _, strategy in sorted(
                enumerate(self._strategies),
                key=lambda x: (x[1].priority, x[0]),
                reverse=True,
            )
        ]
        # Ensure the default strategy is appended last
        strategies.append(self._default_strategy)

        for node in graph_module.graph.nodes:
            for strategy in strategies:
                if isinstance(node.target, str) or node in self._pending_annotate_nodes:
                    continue

                if quant_config := strategy.get_quant_config(node):
                    self._pending_annotate_nodes[node] = (quant_config, strategy)

        if self.verbose:
            print(self.summary())

        for node in graph_module.graph.nodes:
            if isinstance(node.target, str):
                continue
            if node not in self._pending_annotate_nodes:
                print(f"No quant config is implemented for op, {node.target}")
                continue

            OP_ANNOTATOR[node.target](node, self._pending_annotate_nodes[node][0])

        # custom annotation
        self._annotate_custom_annotation(graph_module)

    def add_node_target(
        self,
        targets,
        quant_dtype,
        is_qat,
        act_observer: UniformQuantizationObserverBase,
        granularity: QuantGranularity,
        note: str = "",
        priority: int = 1,
        extra_kwargs: Optional[dict] = None,
    ):
        self._strategies.append(
            ByNodeTarget(
                quant_dtype,
                is_qat,
                granularity,
                act_observer,
                extra_kwargs or {},
                note,
                priority,
                targets,
            ),
        )
        return self

    def add_regex(
        self,
        regex,
        quant_dtype,
        is_qat,
        act_observer: UniformQuantizationObserverBase,
        granularity: QuantGranularity,
        note: str = "",
        priority: int = 1,
        extra_kwargs: Optional[dict] = None,
    ):
        """
        Add a quantization strategy targeting nodes whose module stack matches given regex patterns.

        Args:
            regex (Iterable[str]): Regex patterns to match module stack paths.
            quant_dtype (QuantDtype): Data type for quantization.
            is_qat (bool): Whether to apply QAT or PTQ.
            act_observer (UniformQuantizationObserverBase): Observer for activation quantization.
            granularity (QuantGranularity): Tensor/channel/block granularity.
            note (str): Optional description for the strategy.
            priority (int): Strategy priority (higher value = higher precedence).
            extra_kwargs (dict, optional): Additional parameters for the strategy.
        """
        self._strategies.append(
            ByNameRegex(
                quant_dtype,
                is_qat,
                granularity,
                act_observer,
                extra_kwargs or {},
                note,
                priority,
                regex,
            ),
        )
        return self

    def summary(self, max_rows: int = -1):
        if not self._pending_annotate_nodes:
            return None

        headers = [
            "module_stack",
            "op_target",
            "quantize",
            "act_observer",
            "granularity",
            "note",
            "extra_kwargs",
        ]
        rows = []
        for i, (node, (_, strategy)) in enumerate(self._pending_annotate_nodes.items()):
            if max_rows > 0 and i >= max_rows:
                break

            row = [
                extract_node_metadata_mapping(node),
                node.target,
                f"{strategy.quant_dtype.name}/{'QAT' if strategy.is_qat else 'PTQ'}",
                strategy.act_observer.__name__,
                strategy.granularity.name,
                strategy.note,
                strategy.extra_kwargs,
            ]
            rows.append(row)

        if max_rows > 0 and len(self._pending_annotate_nodes) > max_rows:
            rows.append(["..."] * len(headers))

        return tabulate(rows, headers=headers, tablefmt="grid")
