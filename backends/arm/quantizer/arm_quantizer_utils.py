# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Provide utilities for quantization annotations.

Use these helpers to check and mark annotation state when working with
``QuantizationAnnotation`` entries in FX node metadata.

"""

import logging
import operator
from abc import ABC, abstractmethod
from typing import Any, Callable, cast, Iterator, Optional, TYPE_CHECKING

import torch

from executorch.backends.arm.common.annotation_meta import ArmAnnotationInfo
from executorch.backends.arm.constants import DISALLOW_TFA_META_KEY
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from torch.fx import Node

from torchao.quantization.pt2e.quantizer import (
    DerivedQuantizationSpec,
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
    SharedQuantizationSpec,
)
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from executorch.backends.cortex_m.quantizer.pattern_matcher import PatternMatcher


def is_annotated(node: Node) -> bool:
    """Return True if the node is annotated.

    Args:
        node (Node): FX node to inspect.

    Returns:
        bool: True if ``Q_ANNOTATION_KEY`` exists and ``_annotated`` is set.

    """
    return (
        Q_ANNOTATION_KEY in node.meta
        and cast(QuantizationAnnotation, node.meta[Q_ANNOTATION_KEY])._annotated
    )


def is_output_annotated(node: Node) -> bool:
    """Return True if the node's output is annotated.

    Args:
        node (Node): FX node to inspect.

    Returns:
        bool: True if annotated and an output qspec is present.

    """
    if Q_ANNOTATION_KEY in node.meta:
        annotation = cast(QuantizationAnnotation, node.meta[Q_ANNOTATION_KEY])
        return annotation._annotated and annotation.output_qspec is not None
    else:
        return False


def mark_node_as_annotated(node: Node) -> None:
    """Mark a node as annotated.

    Create an empty ``QuantizationAnnotation`` on the node when missing and set
    its ``_annotated`` flag to True.

    Args:
        node (Node): FX node to update.

    """
    if Q_ANNOTATION_KEY not in node.meta:
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation()
    annotation_info = ArmAnnotationInfo(
        quantized=True,
    )
    node.meta[Q_ANNOTATION_KEY]._annotated = True
    meta_custom = node.meta.get("custom", {})
    meta_custom[ArmAnnotationInfo.CUSTOM_META_KEY] = dict(annotation_info)
    node.meta["custom"] = meta_custom


def has_float_output(node: Node) -> bool:
    meta_val = node.meta.get("val", None)
    if isinstance(meta_val, torch.Tensor):
        return meta_val.dtype.is_floating_point
    return False


def _mark_node_as_quantized(
    node: Node,
    input_qspec_map,
    output_qspec,
    is_quantized,
) -> None:
    """Fill metadata fields used for quantization, partitioning, and
    lowering.
    """
    node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map, output_qspec, _annotated=True
    )

    if node.op == "call_function":
        meta_custom = node.meta.get("custom", {})
        meta_custom[ArmAnnotationInfo.CUSTOM_META_KEY] = ArmAnnotationInfo(
            quantized=is_quantized
        )
        node.meta["custom"] = meta_custom

    node.meta[DISALLOW_TFA_META_KEY] = not is_quantized


def _derive_bias_qparams_fn(
    obs_or_fqs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(obs_or_fqs) != 2:
        raise ValueError(
            f"Expecting two obs/fqs, one for activation and one for weight, got: {len(obs_or_fqs)}"
        )
    act_obs_or_fq = obs_or_fqs[0]
    weight_obs_or_fq = obs_or_fqs[1]
    act_scale, _ = act_obs_or_fq.calculate_qparams()
    weight_scale, _ = weight_obs_or_fq.calculate_qparams()
    return act_scale * weight_scale, torch.full_like(
        weight_scale, fill_value=0, dtype=torch.int32
    )


def _get_int32_bias_qspec(node):
    return DerivedQuantizationSpec(
        derived_from=((node.args[0], node), (node.args[1], node)),  # type: ignore[list-item]
        derive_qparams_fn=_derive_bias_qparams_fn,
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.int32).min,
        quant_max=torch.iinfo(torch.int32).max - 1,
    )


def _get_int32_per_channel_bias_qspec(node):
    return DerivedQuantizationSpec(
        derived_from=((node.args[0], node), (node.args[1], node)),  # type: ignore[list-item]
        derive_qparams_fn=_derive_bias_qparams_fn,
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.int32).min,
        quant_max=torch.iinfo(torch.int32).max - 1,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
    )


class _QuantizerReporterUserMixin:
    def __init__(self):
        self.reporter = None

    def register_reporter(self, reporter) -> None:
        self.reporter = reporter

    def report_reject(self, pattern: list[Node], reason: str) -> None:
        if self.reporter is not None:
            self.reporter.report_reject(self, pattern, reason)

    def report_accept(self, pattern: list[Node]) -> None:
        if self.reporter is not None:
            self.reporter.report_accept(self, pattern)

    def get_quantizer_info(self):
        raise NotImplementedError("Quantizer must implement get_quantizer_info method.")


class PatternCheck:
    """Base class for pattern checks.

    PatternChecks are used to define which patterns are supported for
    quantization and to validate quantization configuration constraints.

    """

    @classmethod
    def is_per_tensor(cls, qspec) -> bool:
        from torchao.quantization.pt2e.quantizer import QuantizationSpecBase

        if not isinstance(qspec, QuantizationSpecBase):
            return False
        return qspec.qscheme in (  # type: ignore[attr-defined]
            torch.per_tensor_affine,
            torch.per_tensor_symmetric,
        )

    @classmethod
    def is_per_channel(cls, qspec) -> bool:
        from torchao.quantization.pt2e.quantizer import QuantizationSpecBase

        if not isinstance(qspec, QuantizationSpecBase):
            return False
        return qspec.qscheme in (  # type: ignore[attr-defined]
            torch.per_channel_affine,
            torch.per_channel_symmetric,
        )

    @classmethod
    def is_int8_activations(
        cls, qconfig: QuantizationConfig, output_node: Node | None = None
    ) -> bool:
        input_qspec = qconfig.get_input_act_qspec()
        output_qspec = qconfig.get_output_act_qspec(output_node)
        from torchao.quantization.pt2e.quantizer import QuantizationSpecBase

        if not isinstance(input_qspec, QuantizationSpecBase) or not isinstance(
            output_qspec, QuantizationSpecBase
        ):
            return False
        return (
            input_qspec.dtype == torch.int8 and output_qspec.dtype == torch.int8  # type: ignore[attr-defined]
        )

    @classmethod
    def check_pattern(cls, pattern: list[Node]) -> bool:
        return True

    @classmethod
    def check_quantization_config(
        cls, pattern: list[Node], quantization_config: QuantizationConfig
    ) -> bool:
        return True


class NodeFinder(ABC):
    @abstractmethod
    def find_nodes(self, model: torch.fx.GraphModule) -> Iterator[Node]:
        """Return nodes of the graph module depending on NodeFinder type.

        Args:
            model (GraphModule): The graph module to search for matching nodes.

        """
        pass


class PatternQuantizer(Quantizer, _QuantizerReporterUserMixin):
    """Quantizes a graph according to an OperatorConfig.

    Args:
        quantization_config (QuantizationConfig): The quantization config to use for annotation.
        node_finder (NodeFinder): The node finder to use for finding nodes to match patterns.
        pattern_matcher (PatternMatcher): The pattern matcher to use for finding patterns in the nodes.

    """

    def __init__(
        self,
        quantization_config: QuantizationConfig | None,
        node_finder: "NodeFinder",
        pattern_matcher: "PatternMatcher",
    ) -> None:
        super().__init__()
        _QuantizerReporterUserMixin.__init__(self)
        self.quantization_config: QuantizationConfig | None = quantization_config
        self.node_finder: "NodeFinder" = node_finder
        self.pattern_matcher: "PatternMatcher" = pattern_matcher

    def get_quantizer_info(self):
        from executorch.backends.cortex_m.quantizer.quantizer_reporter import (
            QuantizerInfo,
            SUPPORTED_QCONFIGS,
        )

        name = self.__class__.__name__
        targeted_nodes_description = str(self.node_finder)
        quantization_config_path = SUPPORTED_QCONFIGS.get(
            self.quantization_config, "UNREGISTERED_QCONFIG"
        )
        support_config_path = self.pattern_matcher.support_dict_name

        return QuantizerInfo(
            name,
            targeted_nodes_description,
            quantization_config_path,
            support_config_path,
        )

    def is_parameter(self, node: Node, model: torch.fx.GraphModule) -> bool:
        """Returns True if the given node is a parameter of the model."""
        try:
            _ = model.get_parameter(node.target)  # type: ignore[arg-type]
            return True
        except Exception:
            return False

    def is_weight(
        self, node: Node, params: list[Node], model: torch.fx.GraphModule
    ) -> bool:
        """Returns True if node is the first parameter of the given
        parameters.
        """
        return len(params) > 0 and node == params[0]

    def is_bias(
        self, node: Node, params: list[Node], model: torch.fx.GraphModule
    ) -> bool:
        """Returns True if node is the second parameter of the given
        parameters.
        """
        return len(params) == 2 and node == params[1]

    def annotate_match(
        self,
        match: list[Node],
        config: QuantizationConfig | None,
        model: torch.fx.GraphModule,
    ) -> None:
        """Annotates a matched pattern according to the given quantization
        config.
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
                if not has_float_output(input_node):
                    continue
                if self.is_weight(input_node, params, model):
                    input_qspec_map[input_node] = (
                        config.get_weight_qspec(node) if config else None
                    )
                elif self.is_bias(input_node, params, model):
                    input_qspec_map[input_node] = (
                        config.get_bias_qspec(node) if config else None  # type: ignore[assignment]
                    )
                elif input_node not in match:
                    input_qspec_map[input_node] = (
                        config.get_input_act_qspec(node, input_node) if config else None
                    )

            if all(node not in match for node in node.users) and output_qspec is None:
                if has_float_output(node):
                    output_qspec = config.get_output_act_qspec(node) if config else None

            _mark_node_as_quantized(
                node,
                input_qspec_map,
                output_qspec,
                config is not None,
            )

    def annotate(self, model: torch.fx.GraphModule) -> None:  # type: ignore[override]
        nodes = self.node_finder.find_nodes(model)
        matches = self.pattern_matcher.find_pattern_matches(
            nodes, self.quantization_config  # type: ignore[arg-type]
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

    def validate(self, model: torch.fx.GraphModule) -> bool:  # type: ignore[override]
        return True


class SharedQspecQuantizer(Quantizer, _QuantizerReporterUserMixin):
    """Assures that specific ops share quantization parameters on all
    inputs/outputs.
    """

    SHARED_QSPEC_OPS_DEFAULT: list[Callable[..., object]] = [
        torch.ops.aten.clone.default,
        torch.ops.aten.lift_fresh_copy.default,
        torch.ops.aten.detach_.default,
        torch.ops.aten.alias.default,
        torch.ops.aten.alias_copy.default,
        torch.ops.aten.copy_.default,
        torch.ops.aten.detach_copy.default,
        torch.ops.aten.unfold_copy.default,
        torch.ops.aten.unbind.int,
        torch.ops.aten.minimum.default,
        torch.ops.aten.maximum.default,
        torch.ops.aten.min.dim,
        torch.ops.aten.max.dim,
        torch.ops.aten.amin.default,
        torch.ops.aten.amax.default,
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
        torch.ops.aten.pad.default,
        torch.ops.aten.constant_pad_nd.default,
        torch.ops.aten.clamp.default,
        torch.ops.aten.clamp.Tensor,
        torch.ops.aten.hardtanh.default,
        torch.ops.aten.hardtanh_.default,
        torch.ops.aten.relu.default,
        torch.ops.aten.relu_.default,
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

    def __init__(self, targets: Optional[list[Callable[..., object]]] = None) -> None:
        super().__init__()
        _QuantizerReporterUserMixin.__init__(self)
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
        from executorch.backends.cortex_m.quantizer.quantizer_reporter import (
            QuantizerInfo,
        )

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

    def _get_input_nodes_with_float_output(self, node: Node) -> list[Node]:
        return [n for n in node.all_input_nodes if has_float_output(n)]

    def _get_user_nodes_with_float_input(self, node: Node) -> list[Node]:
        return [n for n in node.users.keys() if has_float_output(node)]

    def _skip_shared_qspec_from_io(self, node: Node, qspec: QuantizationSpec) -> bool:
        return node.op in ("placeholder", "output") and qspec.dtype == torch.uint8

    def _maybe_enqueue_shared_node(
        self, neighbor: Node, shared_nodes: set[Node], bfs_queue: list[Node]
    ) -> None:
        if neighbor.target in self.targets and neighbor not in shared_nodes:
            if not self._is_annotated(neighbor):
                bfs_queue.append(neighbor)

    def _append_output_qspec(self, node: Node, adjacent_qspecs: list[Any]) -> None:
        if not self._is_annotated(node):
            return
        output_qspec = node.meta.get(  # type: ignore[union-attr]
            Q_ANNOTATION_KEY
        ).output_qspec
        if output_qspec is None:
            return
        if self._skip_shared_qspec_from_io(node, output_qspec):
            return
        adjacent_qspecs.append(output_qspec)

    def _append_input_qspec(
        self, user_node: Node, input_node: Node, adjacent_qspecs: list[Any]
    ) -> None:
        if not self._is_annotated(user_node):
            return
        qspec_map = user_node.meta.get(Q_ANNOTATION_KEY)
        if qspec_map is None:
            return
        if input_node not in qspec_map.input_qspec_map:
            return
        input_qspec = qspec_map.input_qspec_map[input_node]
        if input_qspec is None:
            return
        if self._skip_shared_qspec_from_io(user_node, input_qspec):
            return
        adjacent_qspecs.append(input_qspec)

    def _get_shared_clique(self, root_node: Node) -> tuple[set[Node], list[Any]]:
        shared_nodes = set()
        bfs_queue = [root_node]
        adjacent_qspecs: list[Any] = []

        while bfs_queue:
            node = bfs_queue.pop(0)
            shared_nodes.add(node)

            for input_node in node.all_input_nodes:
                self._maybe_enqueue_shared_node(input_node, shared_nodes, bfs_queue)
                self._append_output_qspec(input_node, adjacent_qspecs)

            for output_node in node.users.keys():
                self._maybe_enqueue_shared_node(output_node, shared_nodes, bfs_queue)
                self._append_input_qspec(output_node, node, adjacent_qspecs)

        return shared_nodes, adjacent_qspecs

    def _annotate_shared_cluster(self, root_node: Node) -> None:
        if (
            len(self._get_input_nodes_with_float_output(root_node)) == 0
            and len(self._get_user_nodes_with_float_input(root_node)) == 0
        ):
            self.report_reject(
                [root_node],
                "No float inputs nor outputs to annotate",
            )
            _mark_node_as_quantized(
                root_node,
                {},
                None,
                is_quantized=True,
            )
            return

        shared_nodes, adjacent_qspecs = self._get_shared_clique(root_node)
        node_order = {node: index for index, node in enumerate(root_node.graph.nodes)}
        ordered_nodes = sorted(shared_nodes, key=lambda node: node_order.get(node, 0))

        if len(adjacent_qspecs) > 0:
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

            shared_qspec = SharedQuantizationSpec((root_node_first_input, root_node))
            for node in shared_nodes:
                input_qspec_map: dict[Node, Optional[QuantizationSpec]] = {
                    n: shared_qspec  # type: ignore[misc]
                    for n in self._get_input_nodes_with_float_output(node)
                }
                if len(self._get_user_nodes_with_float_input(node)) == 0:
                    output_qspec = None
                else:
                    output_qspec = shared_qspec
                _mark_node_as_quantized(
                    node, input_qspec_map, output_qspec, is_quantized=True
                )

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

    def annotate(self, model: torch.fx.GraphModule) -> None:  # type: ignore[override]
        for node in model.graph.nodes:
            if node.target in self.targets and not self._is_annotated(node):
                self._annotate_shared_cluster(node)

    def validate(self, model: torch.fx.GraphModule) -> bool:  # type: ignore[override]
        return True
