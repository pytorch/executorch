# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import operator
from typing import Callable, List, Optional

import torch
from executorch.backends.arm.common.annotation_meta import ArmAnnotationInfo

from executorch.backends.arm.constants import DISALLOW_TFA_META_KEY

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
    INT8_PER_TENSOR_CONFIG,
    QuantizationSpec,
)
from executorch.backends.cortex_m.quantizer.quantizer_reporter import (
    QuantizerInfo,
    QuantizerReporter,
    QuantizerReporterUser,
    SUPPORTED_QCONFIGS,
)
from executorch.backends.cortex_m.quantizer.quantizer_support import (
    __name__ as cortex_m_quantizer_support_module,
    CONV_OP_PATTERNS,
    CONV_TRANSPOSE_OP_PATTERNS,
    CORTEX_M_QUANTIZER_SUPPORT_DICT,
)
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
    input_qspec_map,
    output_qspec,
    is_quantized,
) -> None:
    """Fills various meta data fields required for quantization, partitioning,
    and backend lowering.

    Note: quantization_config is needed to distinguish between otherwise
    identical annotations:
    1. Node explicitly marked as not quantized using quantization_config = None
    2. Node which is quantized but all inputs/outputs are quantized
    """

    # Annotate node for toracho
    node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map, output_qspec, _annotated=True
    )

    # Mark operator nodes as quantized to be folded in fold_qdq_with_annotated_qparams and know what to partition
    if node.op == "call_function":
        meta_custom = node.meta.get("custom", {})
        meta_custom[ArmAnnotationInfo.CUSTOM_META_KEY] = ArmAnnotationInfo(
            quantized=is_quantized
        )
        node.meta["custom"] = meta_custom

    # Mark nodes to not be touched by transform_for_annotation in quantization dry-run
    node.meta[DISALLOW_TFA_META_KEY] = not is_quantized


class CortexMQuantizer(ComposableQuantizer):

    def __init__(self) -> None:
        conv_targets = set()
        for key in CONV_OP_PATTERNS.keys() | CONV_TRANSPOSE_OP_PATTERNS.keys():
            conv_targets.update(key)

        support_dict_name = (
            cortex_m_quantizer_support_module + ".CORTEX_M_QUANTIZER_SUPPORT_DICT"
        )
        pattern_matcher = PatternMatcher(
            CORTEX_M_QUANTIZER_SUPPORT_DICT,
            support_dict_name=support_dict_name,
        )
        quantizers: List[Quantizer] = [
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

    def annotate(self, model):
        reporter = QuantizerReporter(self.quantizers)
        model = super().annotate(model)
        reporter.log_quantizer_report(model)
        return model

    def validate(self, model: GraphModule) -> bool:
        return True

    def transform_for_annotation(self, model: GraphModule) -> GraphModule:
        pass_manager = CortexMPassManager(None)
        return pass_manager.transform_for_annotation(model)


class PatternQuantizer(Quantizer, QuantizerReporterUser):
    """
    Quantizes a graph according to an OperatorConfig.

    Args:
        quantization_config (QuantizationConfig): The quantization config to use for annotation.
        node_finder (NodeFinder): The node finder to use for finding nodes to match patterns.
        pattern_matcher (PatternMatcher): The pattern matcher to use for finding patterns in the nodes.
    """

    def __init__(
        self,
        quantization_config: QuantizationConfig | None,
        node_finder: NodeFinder,
        pattern_matcher: PatternMatcher,
    ) -> None:
        super().__init__()
        self.quantization_config: QuantizationConfig | None = quantization_config
        self.node_finder: NodeFinder = node_finder
        self.pattern_matcher: PatternMatcher = pattern_matcher

    def get_quantizer_info(self):
        name = self.__class__.__name__
        targeted_nodes_description = str(self.node_finder)
        quantization_config_path = SUPPORTED_QCONFIGS.get(
            self.quantization_config, "UNREGISTRED_QCONFIG"
        )
        support_config_path = self.pattern_matcher.support_dict_name

        return QuantizerInfo(
            name,
            targeted_nodes_description,
            quantization_config_path,
            support_config_path,
        )

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
        self, match: List[Node], config: QuantizationConfig | None, model: GraphModule
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
        parameter_targets = {
            torch.ops.aten.linear.default,
            torch.ops.aten.convolution.default,
            torch.ops.aten.conv1d.default,
            torch.ops.aten.conv1d.padding,
            torch.ops.aten.conv2d.default,
            torch.ops.aten.conv2d.padding,
            torch.ops.aten.conv3d.default,
            torch.ops.aten.conv3d.padding,
            torch.ops.aten.conv_transpose2d.input,
        }

        for node in match:
            input_qspec_map = {}
            output_qspec = None

            params = [n for n in node.all_input_nodes if self.is_parameter(n, model)]
            # Check that the assumptions on number of parameters hold to avoid silent errors
            if node.target in parameter_targets:
                if len(params) == 0 or len(params) > 2:
                    logger.warning(
                        f"{node.name} is expected to have parameter tensors for weight/bias but no such inputs found, which may cause unexpected quantization annotations. This is likely caused by incorrect tensor instantiations or non-constant weight/biases."
                    )
            else:
                if len(params) > 0:
                    logger.warning(
                        f"{node.name} is not expected to not have parameter tensors but found {[n.name for n in params]}, which may cause unexpected quantization annotations."
                    )

            for input_node in node.all_input_nodes:
                # Observers only work on floating point tensors, so make sure to skip other dtypes
                if not has_float_output(input_node):
                    continue
                if self.is_weight(input_node, params, model):
                    input_qspec_map[input_node] = (
                        config.get_weight_qspec(node) if config else None
                    )
                elif self.is_bias(input_node, params, model):
                    # Bias qspec is derived from input + weight qspecs
                    input_qspec_map[input_node] = (
                        config.get_bias_qspec(node) if config else None
                    )
                elif input_node not in match:
                    input_qspec_map[input_node] = (
                        config.get_input_act_qspec(node, input_node) if config else None
                    )

            if all(node not in match for node in node.users) and output_qspec is None:
                if has_float_output(node):
                    output_qspec = config.get_output_act_qspec(node) if config else None

            mark_node_as_annotated(
                node,
                input_qspec_map,
                output_qspec,
                config is not None,  # None qconfig -> explicitly not quantized node
            )

    def annotate(self, model: GraphModule) -> None:
        nodes = self.node_finder.find_nodes(model)
        matches = self.pattern_matcher.find_pattern_matches(
            nodes, self.quantization_config
        )
        for result in matches:
            if result.accepted:
                self.annotate_match(result.pattern, self.quantization_config, model)
                self.report_accept(result.pattern)
            else:
                self.report_reject(
                    result.pattern,
                    result.message or "Pattern rejected.",
                )

    def validate(self, model: GraphModule) -> bool:
        return True


class SharedQspecQuantizer(Quantizer, QuantizerReporterUser):
    """
    Special quantizer for assuring that given ops share the same quantization parameters on all input and outputs,
    i.e. ops which does not change the scale such as clone, min/max, transposes and so on.

    Args:
        targets (Optional[List[Callable[..., object]]]): List of operator targets to apply shared
            quantization specs to. If None, a default list of supported ops is used.
    """

    SHARED_QSPEC_OPS_DEFAULT: List[Callable[..., object]] = [
        # Clone
        torch.ops.aten.clone.default,
        torch.ops.aten.lift_fresh_copy.default,
        torch.ops.aten.detach_.default,
        torch.ops.aten.alias.default,
        torch.ops.aten.alias_copy.default,
        torch.ops.aten.copy_.default,
        torch.ops.aten.detach_copy.default,
        torch.ops.aten.unfold_copy.default,
        torch.ops.aten.unbind.int,
        # Min/Max/Mean
        torch.ops.aten.minimum.default,
        torch.ops.aten.maximum.default,
        torch.ops.aten.min.dim,
        torch.ops.aten.max.dim,
        torch.ops.aten.amin.default,
        torch.ops.aten.amax.default,
        # Data shuffling
        torch.ops.aten.permute.default,
        torch.ops.aten.permute_copy.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.transpose_copy.int,
        torch.ops.aten.t_copy.default,
        torch.ops.aten.t.default,
        torch.ops.aten.repeat.default,
        torch.ops.aten.repeat_interleave.self_int,
        torch.ops.aten.expand_copy.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.select.int,
        torch.ops.aten.select_copy.int,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.slice_copy.Tensor,
        torch.ops.aten.split.Tensor,
        torch.ops.aten.split_with_sizes.default,
        torch.ops.aten.split_copy.Tensor,
        torch.ops.aten.tile.default,
        torch.ops.aten.flip.default,
        torch.ops.aten.index_select.default,
        torch.ops.aten.index_put.default,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.as_strided_copy.default,
        torch.ops.aten.pixel_shuffle.default,
        torch.ops.aten.pixel_unshuffle.default,
        torch.ops.aten.cat.default,
        torch.ops.aten.concatenate.default,
        torch.ops.aten.stack.default,
        torch.ops.aten.dropout.default,
        torch.ops.aten.dropout_.default,
        torch.ops.aten.chunk.default,
        torch.ops.aten.index.Tensor,
        torch.ops.aten.gather.default,
        operator.getitem,
        # Change shape
        torch.ops.aten.squeeze.default,
        torch.ops.aten.squeeze_copy.default,
        torch.ops.aten.squeeze_copy.dim,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.squeeze_.dim,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.unsqueeze_copy.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.view.default,
        torch.ops.aten.view_as.default,
        torch.ops.aten.view_copy.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.unflatten.int,
        torch.ops.aten.flatten.using_ints,
        # Padding
        torch.ops.aten.pad.default,
        torch.ops.aten.constant_pad_nd.default,
        # Ativation functions
        torch.ops.aten.clamp.default,
        torch.ops.aten.clamp.Tensor,
        torch.ops.aten.hardtanh.default,
        torch.ops.aten.hardtanh_.default,
        torch.ops.aten.relu.default,
        torch.ops.aten.relu_.default,
        # Logic ops
        torch.ops.aten.eq.Tensor,
        torch.ops.aten.eq.Scalar,
        torch.ops.aten.ne.Tensor,
        torch.ops.aten.ne.Scalar,
        torch.ops.aten.ge.Tensor,
        torch.ops.aten.ge.Scalar,
        torch.ops.aten.gt.Tensor,
        torch.ops.aten.gt.Scalar,
        torch.ops.aten.le.Tensor,
        torch.ops.aten.le.Scalar,
        torch.ops.aten.lt.Tensor,
        torch.ops.aten.lt.Scalar,
        torch.ops.aten.where.self,
        torch.ops.aten.where.default,
        torch.ops.higher_order.while_loop,
        torch.ops.higher_order.cond,
    ]

    def __init__(self, targets: Optional[List[Callable[..., object]]] = None) -> None:
        super().__init__()
        if targets is None:
            self.targets = self.SHARED_QSPEC_OPS_DEFAULT
            self.support_config_path = (
                __name__ + f".{self.__class__.__name__}.SHARED_QSPEC_OPS_DEFAULT"
            )
        else:
            self.targets = targets
            self.support_config_path = (
                f"CUSTOM TARGETS: {', '.join([str(target) for target in targets])}"
            )

    def get_quantizer_info(self):
        name = self.__class__.__name__
        targeted_nodes_description = ""
        quantization_config_path = "SHARED_QCONFIG"
        support_config_path = self.support_config_path
        return QuantizerInfo(
            name,
            targeted_nodes_description,
            quantization_config_path,
            support_config_path,
        )

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
        adjacent_qspecs = []

        while bfs_queue:
            node = bfs_queue.pop(0)
            shared_nodes.add(node)

            # Neighbours may either be other shared nodes, annotated nodes, or non-annotated (float) nodes.
            for input_node in node.all_input_nodes:
                if input_node.target in self.targets and input_node not in shared_nodes:
                    if not self._is_annotated(input_node):
                        bfs_queue.append(input_node)
                if self._is_annotated(input_node):
                    output_qspec = input_node.meta.get(Q_ANNOTATION_KEY).output_qspec
                    if output_qspec is not None:
                        adjacent_qspecs.append(output_qspec)

            for output_node in node.users.keys():
                if (
                    output_node.target in self.targets
                    and output_node not in shared_nodes
                ):
                    if not self._is_annotated(output_node):
                        bfs_queue.append(output_node)
                if (
                    self._is_annotated(output_node)
                    and node in output_node.meta.get(Q_ANNOTATION_KEY).input_qspec_map
                ):
                    input_qspec = output_node.meta.get(
                        Q_ANNOTATION_KEY
                    ).input_qspec_map[node]
                    if input_qspec is not None:
                        adjacent_qspecs.append(input_qspec)

        return shared_nodes, adjacent_qspecs

    def _annotate_shared_cluster(self, root_node: Node) -> None:
        """
        Finds a cluster of unannotated nodes starting in root_node and annotates them with a common
        SharedQuantizationSpec.
        """

        if (
            len(self._get_input_nodes_with_float_output(root_node)) == 0
            and len(self._get_user_nodes_with_float_input(root_node)) == 0
        ):
            self.report_reject(
                [root_node],
                "No float inputs nor outputs to annotate",
            )
            mark_node_as_annotated(
                root_node,
                {},
                None,
                is_quantized=True,
            )
            return

        shared_nodes, adjacent_qspecs = self._get_shared_clique(root_node)
        node_order = {node: index for index, node in enumerate(root_node.graph.nodes)}
        ordered_nodes = sorted(shared_nodes, key=lambda node: node_order.get(node, 0))

        # The selection of root node for the shared_qspec is important for
        # torchao.quantization.pt2e.prepare._create_obs_or_fq_from_qspec:
        # 1. For regular QuantizationSpecs, it creates a new observer
        # 2. For SharedQuantizationSpecs, it returns the observer created for it's root node
        # 3. It handles nodes in the order they appear in graph.nodes
        # This means that we need to make sure that the root node of the shared_qspec
        # has an input node with a quantization spec, so that an observer is created.

        if len(adjacent_qspecs) > 0:
            # Warn if multiple different adjacent qspecs are found.
            if len(adjacent_qspecs) > 1:
                logger.warning(
                    f"Multiple adjacent quantization specs found for {', '.join([n.name for n in ordered_nodes])}, all nodes will share the input quantization spec of {root_node.name}."
                )

            root_node_float_inputs = self._get_input_nodes_with_float_output(root_node)
            if len(root_node_float_inputs) == 0:
                self.report_reject(
                    ordered_nodes,
                    "Couldn't find any floating point input to base shared quantization spec on.",
                )
                return
            root_node_first_input = root_node_float_inputs[0]

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
                mark_node_as_annotated(
                    node, input_qspec_map, output_qspec, is_quantized=True
                )

            # Force the root qspec to be the adjacent spec
            root_node.meta[Q_ANNOTATION_KEY].input_qspec_map[root_node_first_input] = (
                adjacent_qspecs[0]
            )
            self.report_accept(ordered_nodes)

        else:
            self.report_reject(
                ordered_nodes,
                "Couldn't find any adjacent quantization spec to base shared quantization spec on. You may however quantize these nodes manually if required.",
            )
            return

    def annotate(self, model: GraphModule) -> None:
        """
        Annotate shared quantization spec for supported ops.
        """
        for node in model.graph.nodes:
            if node.target in self.targets and not self._is_annotated(node):
                self._annotate_shared_cluster(node)

    def validate(self, model: GraphModule) -> bool:
        return True
