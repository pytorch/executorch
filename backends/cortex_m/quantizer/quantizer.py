# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from collections import defaultdict
from typing import Any, Callable, cast, Iterator, List, Optional

import torch
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm.common.annotation_meta import ArmAnnotationInfo
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.backends.cortex_m.passes.passes_utils import (
    is_channel_broadcast,
    is_channels_last,
)
from executorch.backends.cortex_m.quantizer.node_finders import (
    GlobalNodeFinder,
    NodeFinder,
)
from executorch.backends.cortex_m.quantizer.operator_configs import (
    BINARY_OP_PATTERNS,
    CONV_OP_PATTERNS,
    INT8_BINARY_OPS_OPERATOR_CONFIG,
    INT8_CONV_OPERATOR_CONFIG,
    INT8_CONV_TRANSPOSE_OPERATOR_CONFIG,
    INT8_LINEAR_OPERATOR_CONFIG,
    INT8_SOFTMAX_OPERATOR_CONFIG,
    SOFTMAX_OP_PATTERNS,
)
from executorch.backends.cortex_m.quantizer.quantization_configs import QuantizationSpec
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

    def _transpose_conv_group_filter(self, node: Optional[Node]) -> bool:
        """
        Negative filter function for transpose conv to REJECT:
        1. NCHW memory format (we only support channels_last/NHWC)
        2. Grouped convolutions (groups > 1) - not supported by CMSIS-NN
        3. Non-zero output_padding - not supported by CMSIS-NN
        4. Dilation != 1 - produces incorrect results with CMSIS-NN

        Returns True to REJECT the node, False to ACCEPT.
        """
        if node is None:
            return True  # Reject if node is None

        tensor = get_first_fake_tensor(node)
        if tensor is None:
            return True  # Reject if no tensor found

        # REJECT if using NCHW format (we need channels_last/NHWC)
        if not is_channels_last(tensor):
            return True  # Reject NCHW

        # For aten.conv_transpose2d.input:
        #   (input, weight, bias, stride, padding, output_padding, groups, dilation)
        # Args: 5 = output_padding, 6 = groups, 7 = dilation
        if len(node.args) >= 6:
            output_padding = node.args[5]
            if isinstance(output_padding, (list, tuple)):
                if any(p != 0 for p in output_padding):
                    return True

        if len(node.args) >= 7:
            groups = node.args[6]
            if isinstance(groups, int) and groups > 1:
                return True

        if len(node.args) >= 8:
            dilation = node.args[7]
            if isinstance(dilation, (list, tuple)):
                if any(d != 1 for d in dilation):
                    return True

        return False  # ACCEPT channels_last transpose conv

    @staticmethod
    def _resolve_int(value: Any) -> Optional[int]:
        """Best-effort conversion of FX node arguments to ints."""
        if isinstance(value, int):
            return value
        if hasattr(value, "item"):
            try:
                return int(value.item())  # type: ignore[arg-type]
            except Exception:
                return None
        if hasattr(value, "meta"):
            meta_val = value.meta.get("val")
            return CortexMQuantizer._resolve_int(meta_val)
        return None

    def _extract_dim(self, node: Node) -> Optional[int]:
        """Return the dim argument from a softmax node when statically known."""
        dim_arg = None
        if len(node.args) > 1:
            dim_arg = node.args[1]
        elif "dim" in node.kwargs:
            dim_arg = node.kwargs["dim"]

        if dim_arg is None:
            return -1

        return self._resolve_int(dim_arg)

    def softmax_memory_format_filter(self, node: Optional[Node]) -> bool:
        """
        Return true given the tensor must either
        - be contiguous (default layout) with softmax dim == last logical dim, or
        - be channels_last with softmax dim == channel dim.
        Any other combination is skipped so the op stays in ATen form.
        """
        if node is None:
            return False
        if [node.target] not in SOFTMAX_OP_PATTERNS:
            return False

        tensor = get_first_fake_tensor(node)
        if tensor is None:
            return True

        dim = self._extract_dim(node)
        if dim is None:
            return True

        rank = tensor.dim()
        if rank == 0:
            return True

        positive_dim = dim if dim >= 0 else dim + rank
        if positive_dim < 0 or positive_dim >= rank:
            return True

        is_channels_last = False
        if rank == 4:
            is_channels_last = tensor.is_contiguous(memory_format=torch.channels_last)

        if is_channels_last:
            channel_dim = 1 if rank >= 2 else rank - 1
            if positive_dim != channel_dim:
                return True
        else:
            if positive_dim != rank - 1:
                return True

        return False

    def __init__(self) -> None:
        global_node_finder = GlobalNodeFinder()

        quantizers: List[Quantizer] = [
            OperatorConfigQuantizer(
                INT8_BINARY_OPS_OPERATOR_CONFIG,
                global_node_finder,
                filter_fn=self.broadcasting_filter,
            ),
            OperatorConfigQuantizer(INT8_LINEAR_OPERATOR_CONFIG, global_node_finder),
            OperatorConfigQuantizer(
                INT8_CONV_OPERATOR_CONFIG,
                global_node_finder,
                filter_fn=self.nchw_filter,
            ),
            OperatorConfigQuantizer(
                INT8_CONV_TRANSPOSE_OPERATOR_CONFIG,
                global_node_finder,
                filter_fn=self._transpose_conv_group_filter,
            ),
            OperatorConfigQuantizer(
                INT8_SOFTMAX_OPERATOR_CONFIG,
                global_node_finder,
                filter_fn=self.softmax_memory_format_filter,
            ),
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

    Q_PATTERN_MATCHED_KEY = "quantizer_matched"

    def __init__(
        self,
        operator_config: QuantizationConfig,
        node_finder: NodeFinder,
        filter_fn: Callable[[Node], bool] = lambda node: False,
    ) -> None:
        self.operator_config = operator_config
        self.node_finder = node_finder
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
        self, model: GraphModule, patterns: List[List[OpOverload]]
    ) -> Iterator[List[Node]]:
        """
        Match all given patterns in the graph and return list of matches.
        Each node can only be part of one match, larger patterns are prioritized.
        Currently only linear patterns (single chain) are supported.

        Q_PATTERN_MATCHED_KEY is set to True in node.meta to track which nodes have
        already been matched.
        """

        # maps operator -> list of patterns starting with operator
        patterns_by_first = defaultdict(list)
        for p in sorted(patterns, key=len, reverse=True):
            patterns_by_first[p[0]].append(p)

        for node in self.node_finder.find_nodes(model):
            if node.meta.get(OperatorConfigQuantizer.Q_PATTERN_MATCHED_KEY, False):
                continue
            if node.op == "placeholder" or node.op == "output":
                node.meta["quantizer_matched"] = True
                yield [node]
            for pattern in patterns_by_first.get(node.target, []):
                match_or_none = self.check_pattern(node, pattern)
                if match_or_none is not None:
                    for matched_node in match_or_none:
                        matched_node.meta[
                            OperatorConfigQuantizer.Q_PATTERN_MATCHED_KEY
                        ] = True
                    yield match_or_none

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
        matches = self.match_patterns(model, self.operator_config.operators)
        for match in matches:
            self.annotate_match(match, self.operator_config.config, model)

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
