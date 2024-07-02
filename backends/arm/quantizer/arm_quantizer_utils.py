# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Utility functions for ArmQuantizer
#

from typing import Callable, cast, List

import torch
from torch._subclasses import FakeTensor
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix

from torch.ao.quantization.quantizer import (
    QuantizationAnnotation,
    SharedQuantizationSpec,
)
from torch.fx import GraphModule, Node


def is_annotated(node: Node) -> bool:
    """Given a node return whether the node is annotated."""
    return (
        "quantization_annotation" in node.meta
        and cast(
            QuantizationAnnotation, node.meta["quantization_annotation"]
        )._annotated
    )


def are_annotated(nodes: List[Node]) -> bool:
    """Given a list of nodes (that represents an operator pattern),
    return True if any of the nodes
    is annotated, otherwise return False.
    """
    for node in nodes:
        if is_annotated(node):
            return True
    return False


def mark_nodes_as_annotated(nodes: List[Node]) -> None:
    """Marks all nodes in list 'nodes' as annotated. If needed, an empty
    QuantizationAnnotation is added to the quantization_annotation node meta entry.
    """
    for node in nodes:
        if node is not None:
            if "quantization_annotation" not in node.meta:
                node.meta["quantization_annotation"] = QuantizationAnnotation()
            node.meta["quantization_annotation"]._annotated = True


def is_input_large_scalar(node: Node, gm: GraphModule) -> bool:
    """Check if input is a large scalar value. So that we can skip quantization for the node
    since histc op (in HistogramObserver) only works for values up to certain upper bound
    """
    if node.op == "get_attr" and isinstance(node.target, str):
        tensor = getattr(gm, node.target)
        # torch.histc works until this upper bound
        HISTC_UPPER_BOUND = 3.4028235e15
        return tensor.numel() == 1 and abs(tensor.item()) > HISTC_UPPER_BOUND
    return False


def is_input_non_float_tensor(node: Node) -> bool:
    """Check if the input is not a float tensor, so that we can skip quantization for the node
    since observers only works with float Tensors
    """
    if "val" not in node.meta or not isinstance(node.meta["val"], FakeTensor):
        return True
    return node.meta["val"].dtype != torch.float32


def is_share_obs_or_fq_op(op: Callable) -> bool:
    """Returns whether the the operation 'op' can be quantized using a shared observer or
    fake quantizer. This means that the operation can inherit it's quantization spec
    from parent nodes.
    """
    return op in [
        torch.ops.aten.hardtanh.default,
        torch.ops.aten.hardtanh_.default,
        torch.ops.aten.mean.default,
        torch.ops.aten.mean.dim,
        torch.ops.aten.permute.default,
        torch.ops.aten.permute_copy.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze_copy.dim,
        # TODO: remove?
        torch.ops.aten.adaptive_avg_pool2d.default,
        torch.ops.aten.view_copy.default,
        torch.ops.aten.view.default,
        torch.ops.aten.slice_copy.Tensor,
        torch.ops.aten.flatten.using_ints,
        torch.ops.aten.dropout.default,
    ]


def propagate_annotation(model: GraphModule) -> None:
    """For unannotated ops that can share observer or have fake quantizers,
    annotate with a SharedQuantizationSpec, where the shared spec is the
    output spec of the parent node.
    This propagates output qspecs downward in the graph until
    an op that is already annotated or can't share qspec is encountered.
    """
    for n in model.graph.nodes:
        n = cast(Node, n)
        if is_annotated(n):
            continue
        if n.op != "call_function" or not is_share_obs_or_fq_op(n.target):
            continue

        prev_node = n.args[0]
        if not isinstance(prev_node, Node):
            continue

        quantization_annotation = cast(
            QuantizationAnnotation | None,
            prev_node.meta.get("quantization_annotation", None),
        )
        if not quantization_annotation or not quantization_annotation.output_qspec:
            continue

        # propagate the previous output_qspec to the current node
        shared_qspec = SharedQuantizationSpec(prev_node)
        n.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={
                prev_node: shared_qspec,
            },
            output_qspec=shared_qspec,
            _annotated=True,
        )


def convert_scalars_to_attrs(model: GraphModule) -> GraphModule:
    """For ops in 'targeted_ops', convert inputs that are scalar values
    to attribute Nodes that output the same value.
    #TODO Seems like this should be a pass.
    """
    targeted_ops = [
        torch.ops.aten.add.Tensor,
        torch.ops.aten.mul.Tensor,
    ]
    for n in model.graph.nodes:
        n = cast(Node, n)
        if n.op != "call_function" or n.target not in targeted_ops:
            continue
        args = list(n.args)
        new_args = []
        for i in range(len(args)):
            if isinstance(args[i], Node):
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
