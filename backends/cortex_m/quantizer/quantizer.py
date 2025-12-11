# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable, List, Optional

import torch
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm.common.annotation_meta import ArmAnnotationInfo
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.backends.cortex_m.passes.passes_utils import (
    is_channel_broadcast,
    is_channels_last,
)
from executorch.backends.cortex_m.quantizer.operator_configs import (
    BINARY_OP_PATTERNS,
    CONV_OP_PATTERNS,
    INT8_BINARY_OPS_OPERATOR_CONFIG,
    INT8_CONV_OPERATOR_CONFIG,
    INT8_LINEAR_OPERATOR_CONFIG,
)
from executorch.backends.cortex_m.quantizer.quantization_configs import (
    INT8_PER_TENSOR_CONFIG,
    QuantizationSpec,
)
from torch._ops import OpOverload
from torch.fx import GraphModule, Node
from torchao.quantization.pt2e.quantizer import (
    ComposableQuantizer,
    QuantizationAnnotation,
    Quantizer,
    SharedQuantizationSpec,
)
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY


def mark_node_as_annotated(
    node: Node,
    input_qspec_map: dict[Node, Optional[QuantizationSpec]],
    output_qspec: Optional[QuantizationSpec],
) -> None:
    node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(input_qspec_map, output_qspec)
    annotation_info = ArmAnnotationInfo(
        quantized=True,
    )
    meta_custom = node.meta.get("custom", {})
    meta_custom[ArmAnnotationInfo.CUSTOM_META_KEY] = dict(annotation_info)
    node.meta["custom"] = meta_custom


class CortexMQuantizer(ComposableQuantizer):

    def broadcasting_filter(self, node: Optional[Node]) -> bool:
        """
        Filter function to exclude nodes that perform broadcasting.
        """
        if node is None:
            return False
        if [node.target] not in BINARY_OP_PATTERNS:
            return False

        if len(node.all_input_nodes) == 2:
            t1 = get_first_fake_tensor(node.all_input_nodes[0])
            t2 = get_first_fake_tensor(node.all_input_nodes[1])
            return t1.shape != t2.shape and not (
                is_channel_broadcast(t1, t2) and is_channels_last(t1)
            )

        return False

    def nchw_filter(self, node: Optional[Node]) -> bool:
        """
        Filter function to exclude nodes that use NCHW memory format.
        """
        if node is None:
            return False
        if [node.target] not in CONV_OP_PATTERNS:
            return False

        tensor = get_first_fake_tensor(node)
        if tensor is None:
            return False

        return not is_channels_last(tensor)

    def __init__(self) -> None:
        quantizers: List[Quantizer] = [
            OperatorConfigQuantizer(
                INT8_BINARY_OPS_OPERATOR_CONFIG, filter_fn=self.broadcasting_filter
            ),
            OperatorConfigQuantizer(INT8_LINEAR_OPERATOR_CONFIG),
            OperatorConfigQuantizer(
                INT8_CONV_OPERATOR_CONFIG, filter_fn=self.nchw_filter
            ),
            InputQuantizer(INT8_PER_TENSOR_CONFIG),
            OutputQuantizer(INT8_PER_TENSOR_CONFIG),
            SharedQspecQuantizer(),
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

            if all(node not in match for node in node.users) and output_qspec is None:
                output_qspec = config.output_activation if config else None

            mark_node_as_annotated(node, input_qspec_map, output_qspec)

    def annotate(self, model: GraphModule) -> None:
        matches = self.match_patterns(model, self.operator_config.operators)
        for match in matches:
            self.annotate_match(match, self.operator_config.config, model)

    def validate(self, model: GraphModule) -> bool:
        return True


class InputQuantizer(Quantizer):
    """
    Quantizes only the input activations of the graph.
    """

    def __init__(
        self,
        quantization_config: QuantizationConfig,
        filter_fn: Callable[[Node], bool] = lambda node: False,
    ) -> None:
        self.quantization_config = quantization_config
        self.filter_fn = filter_fn

    def annotate(self, model: GraphModule) -> None:
        for node in model.graph.nodes:
            is_placeholder = node.op == "placeholder"
            is_filtered = self.filter_fn(node)
            if is_placeholder and not is_filtered:
                mark_node_as_annotated(
                    node, {}, self.quantization_config.output_activation
                )

    def validate(self, model: GraphModule) -> bool:
        return True


class OutputQuantizer(Quantizer):
    """
    Quantizes only the output activations of the graph.
    """

    def __init__(
        self,
        quantization_config: QuantizationConfig,
        filter_fn: Callable[[Node], bool] = lambda node: False,
    ) -> None:
        self.quantization_config = quantization_config
        self.filter_fn = filter_fn

    def annotate(self, model: GraphModule) -> None:
        output_node = model.graph.output_node()
        input_qspec_map = {
            n: self.quantization_config.input_activation
            for n in output_node.all_input_nodes
            if not self.filter_fn(n)
        }
        output_qspec = self.quantization_config.output_activation
        mark_node_as_annotated(output_node, input_qspec_map, output_qspec)

    def validate(self, model: GraphModule) -> bool:
        return True


class SharedQspecQuantizer(Quantizer):
    """
    Special quantizer for assuring that given ops share the same quantization parameters on all input and outputs,
    i.e. ops which does not change the scale such as clone, min/max, transposes and so on.

    Args:
        targets (Optional[List[OpOverload]]): List of operator overloads to apply shared quantization spec to.
            If None, a default list of supported ops is used.
    """

    SHARED_QSPEC_OPS_DEFAULT: List[OpOverload] = [
        # Clone
        torch.ops.aten.clone.default,
        torch.ops.aten.lift_fresh_copy.default,
        torch.ops.aten.detach_.default,
        # Min/Max/Mean
        torch.ops.aten.minimum.default,
        torch.ops.aten.maximum.default,
        # Data shuffling
        torch.ops.aten.permute.default,
        torch.ops.aten.permute_copy.default,
        torch.ops.aten.transpose.Dimname,
        torch.ops.aten.transpose.int,
        torch.ops.aten.transpose_copy.int,
        torch.ops.aten.t_copy.default,
        torch.ops.aten.t.default,
        # Change shape
        torch.ops.aten.squeeze.default,
        torch.ops.aten.squeeze_copy.default,
        torch.ops.aten.squeeze_copy.dim,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.unsqueeze_copy.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.view.default,
        torch.ops.aten.view_as.default,
        torch.ops.aten.view_copy.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.unflatten.int,
        torch.ops.aten.flatten.using_ints,
    ]

    def __init__(self, targets: Optional[List[OpOverload]] = None) -> None:
        super().__init__()
        if targets is None:
            self.targets = self.SHARED_QSPEC_OPS_DEFAULT
        else:
            self.targets = targets

    def _is_annotated(self, node: Node) -> bool:
        return Q_ANNOTATION_KEY in node.meta

    def _annotate_shared_cluster(self, root_node: Node) -> None:
        """
        Finds a cluster of unannotated nodes starting in root_node and annotates them with a common
        SharedQuantizationSpec.
        """

        shared_nodes = set()
        leaf_nodes = set()
        bfs_queue = [root_node]

        while bfs_queue:
            node = bfs_queue.pop(0)

            if self._is_annotated(node):
                leaf_nodes.add(node)
                continue
            if node.op == "get_attr":
                continue

            if node.target not in self.targets:
                raise NotImplementedError(
                    (
                        f"{SharedQspecQuantizer.__name__} found unannoted node '{node.name}' in neighbour_nodes "
                        "which is not in the supported target list. This might be the case either because:\n"
                        "1) The op should have shared qspec but is not in the target list. "
                        "In this case, try modifying the list using the targets field in the initializer.\n"
                        "2) The op should not be quantized, which is not currently supported by the SharedQspecQuantizer."
                    )
                )

            shared_nodes.add(node)
            neighbour_nodes = list(node.all_input_nodes) + list(node.users)
            for n in neighbour_nodes:
                if n not in shared_nodes:
                    bfs_queue.append(n)

        # The selection of root node for the shared_qspec is important for
        # torchao.quantization.pt2e.prepare._create_obs_or_fq_from_qspec:
        # 1. For regular QuantizationSpecs, it creates a new observer
        # 2. For SharedQuantizationSpecs, it returns the observer created for it's root node
        # 3. It handles nodes in the order they appear in graph.nodes
        # This means that the root node of the shared group needs to be the first annotated node that appears in graph.nodes.
        shared_root_node = next(n for n in root_node.graph.nodes if n in leaf_nodes)
        shared_qspec = SharedQuantizationSpec(shared_root_node)

        for node in shared_nodes:
            input_qspec_map: dict[Node, Optional[QuantizationSpec]] = {
                n: shared_qspec for n in node.all_input_nodes
            }
            mark_node_as_annotated(node, input_qspec_map, shared_qspec)

    def annotate(self, model: GraphModule) -> None:
        for node in model.graph.nodes:
            if node.target in self.targets and not self._is_annotated(node):
                self._annotate_shared_cluster(node)

    def validate(self, model: GraphModule) -> bool:
        return True
