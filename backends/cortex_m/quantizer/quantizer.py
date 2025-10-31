# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable, List, Optional

import torch

from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor

from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.backends.cortex_m.quantizer.operator_configs import (
    INT8_BINARY_OPS_OPERATOR_CONFIG,
    INT8_LINEAR_OPERATOR_CONFIG,
)
from torch._ops import OpOverload
from torch.fx import GraphModule, Node
from torchao.quantization.pt2e.quantizer import (
    ComposableQuantizer,
    QuantizationAnnotation,
    Quantizer,
)
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY


class CortexMQuantizer(ComposableQuantizer):

    def broadcasting_filter(self, node: Optional[Node]) -> bool:
        """
        Filter function to exclude nodes that perform broadcasting.
        """
        if node is None:
            return False
        if node.target not in [torch.ops.aten.add.Tensor]:
            return False

        if len(node.all_input_nodes) == 2:
            t1 = get_first_fake_tensor(node.all_input_nodes[0])
            t2 = get_first_fake_tensor(node.all_input_nodes[1])
            return t1.shape != t2.shape

        return False

    def __init__(self) -> None:
        quantizers: List[OperatorConfigQuantizer] = [
            OperatorConfigQuantizer(
                INT8_BINARY_OPS_OPERATOR_CONFIG, filter_fn=self.broadcasting_filter
            ),
            OperatorConfigQuantizer(INT8_LINEAR_OPERATOR_CONFIG),
        ]
        super().__init__(quantizers)

    def validate(self, model: GraphModule) -> bool:
        return True

    def transform_for_annotation(self, model: GraphModule) -> GraphModule:
        pass_manager = CortexMPassManager(None)
        return pass_manager.transform_for_annotation(model)


class OperatorConfigQuantizer(Quantizer):
    """
    Quantizes a graph according to an OperatorConfig.

    Args:
        operator_config (OperatorConfig): The operator config to use for quantization.
        filter_fn (Callable): Negative filter function. If it returns True on any node in the pattern, the pattern is
                              skipped. Used to match for example particular targets or modules.
    """

    def __init__(
        self,
        operator_config: QuantizationConfig,
        filter_fn: Callable[[Node], bool] = lambda node: False,
    ) -> None:
        self.operator_config = operator_config
        self.filter_fn = filter_fn

    def check_node(self, node: Optional[Node], target: str) -> bool:
        """
        Return true if the node is a valid match for the given target.
        """
        if node is None:
            return False
        if not node.target == target:
            return False
        if node.meta.get("quantizer_matched", False):
            return False
        if self.filter_fn(node):
            return False

        return True

    def check_pattern(
        self, node: Optional[Node], pattern: List[OpOverload]
    ) -> Optional[List[Node]]:
        """
        Returns the matched nodes if the given node matches the given pattern, otherwise None.
        """
        match: List[Node] = []
        node = list(node.users)[0] if node and len(node.users) > 0 else None

        for pattern_target in pattern:
            if self.check_node(node, pattern_target):
                match.append(node)
                node = list(node.users)[0] if len(node.users) > 0 else None
            else:
                return None

        return match

    def match_patterns(
        self, model: GraphModule, patterns: List[List[str]]
    ) -> List[List[Node]]:
        """
        Match all given patterns in the graph and return list of matches.
        Each node can only be part of one match, larger patterns are prioritized.
        Currently only linear patterns (single chain) are supported.
        """
        patterns.sort(key=len, reverse=True)
        matches: List[List[Node]] = []
        for pattern in patterns:
            for node in model.graph.nodes:
                potential_match = self.check_pattern(node, pattern)
                if potential_match:
                    matches.append(potential_match)
                    for node in potential_match:
                        node.meta["quantizer_matched"] = True

        return matches

    def is_parameter(self, node: Node, model: GraphModule) -> bool:
        """Returns True if the given node is a parameter of the model."""
        try:
            _ = model.get_parameter(node.target)
            return True
        except Exception:
            return False

    def is_weight(self, node: Node, params: List[Node], model: GraphModule) -> bool:
        """Returns True if node is the first parameter of the given parameters"""
        return len(params) > 0 and node == params[0]

    def is_bias(self, node: Node, params: List[Node], model: GraphModule) -> bool:
        """Returns True if node is the second parameter of the given parameters"""
        return len(params) == 2 and node == params[1]

    def annotate_match(
        self, match: List[Node], config: QuantizationConfig, model: GraphModule
    ) -> None:
        """
        Annotates a matched pattern according to the given quantization config. The
        following assumptions are made:

        - All operators have either no parameters, only weights, or weights and biases
        - Tensors which are the first parameter of an operator are annotated as weights
        - Tensors which are the second parameter of an operator are annotated as biases
        - All other tensors going into the matched pattern are annotated as input activations.
        - All other outputs coming out of the matched pattern are annotated as output activations.

        """
        for node in match:
            input_qspec_map = {}
            output_qspec = None

            params = [n for n in node.all_input_nodes if self.is_parameter(n, model)]
            # Check that the assumptions on number of parameters hold to avoid silent errors
            assert (
                0 <= len(params) <= 2
            ), f"{self.__class__.__name__} expected 0 params, 1 params (weight) or 2 params (weight, bias), but got {len(params)} for node {node}."

            for input_node in node.all_input_nodes:
                if self.is_weight(input_node, params, model):
                    input_qspec_map[input_node] = config.weight if config else None
                elif self.is_bias(input_node, params, model):
                    # Bias qspec is derived from input + weight qspecs
                    input_qspec_map[input_node] = config.bias(node) if config else None
                elif input_node not in match:
                    input_qspec_map[input_node] = (
                        config.input_activation if config else None
                    )

            if all(node not in match for node in node.users):
                output_qspec = config.output_activation if config else None

            node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map, output_qspec
            )

    def annotate(self, model: GraphModule) -> None:
        matches = self.match_patterns(model, self.operator_config.operators)
        for match in matches:
            self.annotate_match(match, self.operator_config.config, model)

    def validate(self, model: GraphModule) -> bool:
        return True
