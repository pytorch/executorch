# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Callable, Optional, Tuple

import torch
from torch.fx import Node
from torchao.quantization.pt2e.quantizer import (
    annotate_input_qspec_map,
    annotate_output_qspec,
    get_bias_qspec,
    get_input_act_qspec,
    get_output_act_qspec,
    get_weight_qspec,
    QuantizationAnnotation,
    QuantizationConfig,
    SharedQuantizationSpec,
)
from torchao.quantization.pt2e.utils import get_new_attr_name_with_prefix

__all__ = [
    "OP_TO_ANNOTATOR",
    "propagate_annotation",
    "_convert_scalars_to_attrs",
    "bits_to_range",
]


def bits_to_range(bits: int) -> Tuple[int, int]:
    """
    Calculate quantization range for given number of bits.

    Args:
        bits: Number of quantization bits

    Returns:
        Tuple of (qmin, qmax) for the given bit width
    """
    return (
        -(2 ** (bits - 1)),
        (2 ** (bits - 1) - 1),
    )


AnnotatorType = Callable[
    [
        torch.fx.GraphModule,
        Optional[QuantizationConfig],
        Optional[Callable[[Node], bool]],
    ],
    Optional[list[list[Node]]],
]
OP_TO_ANNOTATOR: dict[str, AnnotatorType] = {}


def register_annotator(op: str) -> Callable[[AnnotatorType], None]:
    def decorator(annotator: AnnotatorType) -> None:
        OP_TO_ANNOTATOR[op] = annotator

    return decorator


def _is_annotated(nodes: list[Node]) -> bool:
    """
    Given a list of nodes (that represents an operator pattern),
    check if any of the node is annotated, return True if any of the node
    is annotated, otherwise return False
    """
    annotated = False
    for node in nodes:
        annotated = annotated or (
            "quantization_annotation" in node.meta
            and node.meta["quantization_annotation"]._annotated
        )
    return annotated


def _mark_nodes_as_annotated(nodes: list[Node]) -> None:
    for node in nodes:
        if node is not None:
            if "quantization_annotation" not in node.meta:
                node.meta["quantization_annotation"] = QuantizationAnnotation()
            node.meta["quantization_annotation"]._annotated = True


@register_annotator("linear")
def _annotate_linear(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    annotated_partitions = []
    input_act_qspec = get_input_act_qspec(quantization_config)
    output_act_qspec = get_output_act_qspec(quantization_config)
    weight_qspec = get_weight_qspec(quantization_config)
    bias_qspec = get_bias_qspec(quantization_config)
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.linear.default:
            continue
        if filter_fn and not filter_fn(node):
            continue
        act_node = node.args[0]
        weight_node = node.args[1]
        bias_node = None
        if len(node.args) > 2:
            bias_node = node.args[2]

        if _is_annotated([node]) is False:  # type: ignore[list-item]
            annotate_input_qspec_map(
                node,
                act_node,
                input_act_qspec,
            )
            annotate_input_qspec_map(
                node,
                weight_node,
                weight_qspec,
            )
            nodes_to_mark_annotated = [node, weight_node]
            if bias_node:
                annotate_input_qspec_map(
                    node,
                    bias_node,
                    bias_qspec,
                )
                nodes_to_mark_annotated.append(bias_node)
            annotate_output_qspec(node, output_act_qspec)
            _mark_nodes_as_annotated(nodes_to_mark_annotated)
            annotated_partitions.append(nodes_to_mark_annotated)

    return annotated_partitions


def _is_share_obs_or_fq_op(op: Callable[..., torch.Tensor]) -> bool:
    return op in [
        torch.ops.aten.relu.default,
        torch.ops.aten.hardtanh.default,
        torch.ops.aten.hardtanh_.default,
        torch.ops.aten.max_pool2d.default,
        torch.ops.aten.mean.default,
        torch.ops.aten.mean.dim,
        torch.ops.aten.permute.default,
        torch.ops.aten.permute_copy.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze_copy.dim,
        torch.ops.aten.adaptive_avg_pool2d.default,
        torch.ops.aten.view_copy.default,
        torch.ops.aten.view.default,
        torch.ops.aten.slice_copy.Tensor,
        torch.ops.aten.flatten.using_ints,
    ]


def propagate_annotation(model: torch.fx.GraphModule) -> None:
    for n in model.graph.nodes:
        if n.op != "call_function" or not _is_share_obs_or_fq_op(n.target):
            continue

        prev_node = n.args[0]
        if not isinstance(prev_node, Node):
            continue

        quantization_annotation = prev_node.meta.get("quantization_annotation", None)
        if not quantization_annotation:
            continue

        output_qspec = quantization_annotation.output_qspec
        if not output_qspec:
            continue

        # make sure current node is not annotated
        if (
            "quantization_annotation" in n.meta
            and n.meta["quantization_annotation"]._annotated
        ):
            continue

        shared_qspec = SharedQuantizationSpec(prev_node)
        # propagate the previous output_qspec to the current node
        n.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={
                prev_node: shared_qspec,
            },
            output_qspec=shared_qspec,
            _annotated=True,
        )


def _convert_scalars_to_attrs(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for n in model.graph.nodes:
        if n.op != "call_function" or n.target not in [
            torch.ops.aten.add.Tensor,
            torch.ops.aten.mul.Tensor,
        ]:
            continue
        args = list(n.args)
        new_args = []
        for i in range(len(args)):
            if isinstance(args[i], torch.fx.Node):
                new_args.append(args[i])
                continue
            prefix = "_tensor_constant_"
            get_new_attr_name = get_new_attr_name_with_prefix(prefix)
            tensor_constant_name = get_new_attr_name(model)
            float_tensor = torch.tensor(float(args[i]))
            model.register_buffer(tensor_constant_name, float_tensor)
            fake_mode = n.meta["val"].fake_mode
            with model.graph.inserting_before(n):
                get_attr_node = model.graph.create_node(
                    "get_attr", tensor_constant_name, (), {}
                )
                get_attr_node.meta["val"] = fake_mode.from_tensor(
                    float_tensor, static_shapes=True
                )
                new_args.append(get_attr_node)
        n.args = tuple(new_args)
    model.recompile()
    return model
