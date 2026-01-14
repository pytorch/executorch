# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import cast, List, Optional

import torch
from executorch.backends.arm.common.annotation_meta import ArmAnnotationInfo
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.backends.cortex_m.quantizer.node_finders import (
    GlobalNodeFinder,
    NodeFinder,
    NodeTargetNodeFinder,
)
from executorch.backends.cortex_m.quantizer.pattern_matcher import PatternMatcher
from executorch.backends.cortex_m.quantizer.quantization_configs import (
    INT8_PER_CHANNEL_CONFIG,
    INT8_PER_CHANNEL_TRANSPOSE_CONFIG,
    INT8_PER_TENSOR_CONFIG,
    QuantizationSpec,
    SOFTMAX_PER_TENSOR_CONFIG,
)
from executorch.backends.cortex_m.quantizer.quantizer_support import (
    CONV_OP_PATTERNS,
    CONV_TRANSPOSE_OP_PATTERNS,
    CORTEX_M_QUANTIZER_SUPPORT_DICT,
    SOFTMAX_OP_PATTERNS,
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

logger = logging.getLogger(__name__)


def has_float_output(node: Node) -> bool:
    meta_val = node.meta.get("val", None)
    if isinstance(meta_val, torch.Tensor):
        return meta_val.dtype.is_floating_point

    return False


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

    def __init__(self) -> None:
        conv_targets = set()
        for key in CONV_OP_PATTERNS.keys():
            conv_targets.update(key)

        softmax_targets = set()
        for key in SOFTMAX_OP_PATTERNS.keys():
            softmax_targets.update(key)

        conv_transpose_targets = set()
        for key in CONV_TRANSPOSE_OP_PATTERNS:
            conv_transpose_targets.update(key)

        pattern_matcher = PatternMatcher(CORTEX_M_QUANTIZER_SUPPORT_DICT)
        quantizers: List[Quantizer] = [
            PatternQuantizer(
                SOFTMAX_PER_TENSOR_CONFIG,
                node_finder=NodeTargetNodeFinder(softmax_targets),
                pattern_matcher=pattern_matcher,
            ),
            PatternQuantizer(
                INT8_PER_CHANNEL_TRANSPOSE_CONFIG,
                node_finder=NodeTargetNodeFinder(conv_transpose_targets),
                pattern_matcher=pattern_matcher,
            ),
            PatternQuantizer(
                INT8_PER_CHANNEL_CONFIG,
                node_finder=NodeTargetNodeFinder(conv_targets),
                pattern_matcher=pattern_matcher,
            ),
            PatternQuantizer(
                INT8_PER_TENSOR_CONFIG,
                node_finder=GlobalNodeFinder(),
                pattern_matcher=pattern_matcher,
            ),
            SharedQspecQuantizer(),
        ]
        super().__init__(quantizers)

    def validate(self, model: GraphModule) -> bool:
        return True

    def transform_for_annotation(self, model: GraphModule) -> GraphModule:
        pass_manager = CortexMPassManager(None)
        return pass_manager.transform_for_annotation(model)


class PatternQuantizer(Quantizer):
    """
    Quantizes a graph according to an OperatorConfig.

    Args:
        quantization_config (QuantizationConfig): The quantization config to use for annotation.
        node_finder (NodeFinder): The node finder to use for finding nodes to match patterns.
        pattern_matcher (PatternMatcher): The pattern matcher to use for finding patterns in the nodes.
    """

    def __init__(
        self,
        quantization_config: QuantizationConfig,
        node_finder: NodeFinder,
        pattern_matcher: PatternMatcher,
    ) -> None:
        self.quantization_config = quantization_config
        self.node_finder = node_finder
        self.pattern_matcher = pattern_matcher

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
                # Observers only work on floating point tensors, so make sure to skip other dtypes
                if not has_float_output(input_node):
                    continue
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
        nodes = self.node_finder.find_nodes(model)
        matches = self.pattern_matcher.find_pattern_matches(
            nodes, self.quantization_config
        )
        for result in matches:
            if result.accepted:
                self.annotate_match(result.pattern, self.quantization_config, model)
            else:
                logger.debug(
                    f"PatternQuantizer skipped annotation of pattern {[n.target for n in result.pattern]}: {result.message}"
                )

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
        torch.ops.aten.avg_pool2d.default,
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

    def _get_input_nodes_with_float_output(self, node: Node) -> List[Node]:
        # Observers only work on floating point tensors, so make sure to skip other dtypes
        return [n for n in node.all_input_nodes if has_float_output(n)]

    def _get_user_nodes_with_float_input(self, node: Node) -> List[Node]:
        # Observers only work on floating point tensors, so make sure to skip other dtypes
        return [n for n in node.users.keys() if has_float_output(node)]

    def _get_shared_clique(self, root_node: Node) -> set[Node]:
        """
        Finds a cluster of nodes with targets in self.targets, starting in root_node.
        """
        shared_nodes = set()
        bfs_queue = [root_node]
        adjacent_qspecs = set()

        while bfs_queue:
            node = bfs_queue.pop(0)
            shared_nodes.add(node)

            # Neighbours may either be other shared nodes, annotated nodes, or non-annotated (float) nodes.
            for input_node in self._get_input_nodes_with_float_output(node):
                if input_node.target in self.targets and input_node not in shared_nodes:
                    if not self._is_annotated(input_node):
                        bfs_queue.append(input_node)
                if self._is_annotated(input_node):
                    output_qspec = input_node.meta.get(
                        Q_ANNOTATION_KEY, None
                    ).output_qspec
                    adjacent_qspecs.add(output_qspec)

            for output_node in self._get_user_nodes_with_float_input(node):
                if (
                    output_node.target in self.targets
                    and output_node not in shared_nodes
                ):
                    if not self._is_annotated(output_node):
                        bfs_queue.append(output_node)
                if self._is_annotated(output_node):
                    input_qspec = output_node.meta.get(
                        Q_ANNOTATION_KEY, None
                    ).input_qspec_map[node]
                    adjacent_qspecs.add(input_qspec)

        return shared_nodes, adjacent_qspecs

    def _annotate_shared_cluster(self, root_node: Node) -> None:
        """
        Finds a cluster of unannotated nodes starting in root_node and annotates them with a common
        SharedQuantizationSpec.
        """

        shared_nodes, adjacent_qspecs = self._get_shared_clique(root_node)

        # The selection of root node for the shared_qspec is important for
        # torchao.quantization.pt2e.prepare._create_obs_or_fq_from_qspec:
        # 1. For regular QuantizationSpecs, it creates a new observer
        # 2. For SharedQuantizationSpecs, it returns the observer created for it's root node
        # 3. It handles nodes in the order they appear in graph.nodes
        # This means that we need to make sure that the root node of the shared_qspec
        # has an input node with a quantization spec, so that an observer is created.

        if len(adjacent_qspecs) == 1:
            root_node_first_input = self._get_input_nodes_with_float_output(root_node)[
                0
            ]

            # Make all nodes share qspec with the root node's first input
            shared_qspec = SharedQuantizationSpec((root_node_first_input, root_node))
            for node in shared_nodes:
                input_qspec_map: dict[Node, Optional[QuantizationSpec]] = {
                    n: shared_qspec
                    for n in self._get_input_nodes_with_float_output(node)
                }
                if len(self._get_user_nodes_with_float_input(node)) == 0:
                    output_qspec = None
                else:
                    output_qspec = shared_qspec
                mark_node_as_annotated(node, input_qspec_map, output_qspec)

            # Force the root qspec to be the adjacent spec
            root_node.meta[Q_ANNOTATION_KEY].input_qspec_map[
                root_node_first_input
            ] = adjacent_qspecs.pop()

        elif len(adjacent_qspecs) == 0:
            logger.warning(
                "SharedQspecQuantizer found a cluster of supported ops surrounded by no quantized ops - leaving nodes unquantized."
            )
            return
        else:
            logger.warning(
                "SharedQspecQuantizer found a cluster of supported ops surrounded by multiple different qspecs - leaving nodes unquantized."
            )
            return

    def annotate(self, model: GraphModule) -> None:
        """
        Annotate shared quantization spec for supported ops, but skip avg_pool2d
        when both ceil_mode and count_include_pad are True.
        """
        for node in model.graph.nodes:
            # TODO Skip avg_pool2d when ceil_mode=True or count_include_pad=True
            # CMSIS-NN doesn't directly support this. But, it should be done.
            if node.target is torch.ops.aten.avg_pool2d.default:
                ceil_mode = cast(bool, node.args[4]) if len(node.args) > 4 else False
                count_include_pad = (
                    cast(bool, node.args[5]) if len(node.args) > 5 else True
                )
                if ceil_mode or count_include_pad:
                    continue
            if node.target in self.targets and not self._is_annotated(node):
                self._annotate_shared_cluster(node)

    def validate(self, model: GraphModule) -> bool:
        return True
