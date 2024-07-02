# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Utility functions for ArmQuantizer
#

from typing import Callable, List

import torch
import torch.nn.functional as F
from torch._subclasses import FakeTensor
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from torch.ao.quantization.pt2e.export_utils import _WrapperModule

from torch.ao.quantization.quantizer import (
    QuantizationAnnotation,
    SharedQuantizationSpec,
)
from torch.fx import Node


def is_annotated(nodes: List[Node]):
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


def mark_nodes_as_annotated(nodes: List[Node]):
    for node in nodes:
        if node is not None:
            if "quantization_annotation" not in node.meta:
                node.meta["quantization_annotation"] = QuantizationAnnotation()
            node.meta["quantization_annotation"]._annotated = True


def get_pattern(conv_fn: Callable, relu_is_inplace: bool, has_relu: bool):
    def _conv_bn(x, conv_weight, conv_bias, bn_weight, bn_bias, bn_rm, bn_rv):
        conv = conv_fn(x, conv_weight, conv_bias)
        bn = F.batch_norm(conv, bn_rm, bn_rv, bn_weight, bn_bias, training=True)
        if has_relu:
            output = F.relu_(bn) if relu_is_inplace else F.relu(bn)
        else:
            output = bn
        return output, {
            "input": x,
            "conv": conv,
            "weight": conv_weight,
            "bias": conv_bias,
            "output": output,
        }

    return _WrapperModule(_conv_bn)


def is_input_large_scalar(node: Node, gm: torch.fx.GraphModule):
    """Check if input is a large scalar value. So that we can skip quantization for the node
    since histc op (in HistogramObserver) only works for values up to certain upper bound
    """
    if node.op == "get_attr":
        tensor = getattr(gm, node.target)  # type: ignore[arg-type]
        # torch.histc works until this upper bound
        HISTC_UPPER_BOUND = 3.4028235e15
        return tensor.numel() == 1 and abs(tensor.item()) > HISTC_UPPER_BOUND
    return False


def is_input_non_float_tensor(node: Node):
    """Check if the input is not a float tensor, so that we can skip quantization for the node
    since observers only works with float Tensors
    """
    if "val" not in node.meta or not isinstance(node.meta["val"], FakeTensor):
        return True
    return node.meta["val"].dtype != torch.float32


def is_share_obs_or_fq_op(op: Callable) -> bool:
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
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.flatten.using_ints,
        torch.ops.aten.dropout.default,
    ]


def propagate_annotation(model: torch.fx.GraphModule) -> None:
    for n in model.graph.nodes:
        if n.op != "call_function" or not is_share_obs_or_fq_op(n.target):
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


# TODO: make the list of ops customizable
def convert_scalars_to_attrs(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
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
