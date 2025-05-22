# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

#
# Utility functions for TOSAQuantizer
#

from typing import cast, Sequence

import torch
from torch._subclasses import FakeTensor
from torch.fx import GraphModule, Node

from torchao.quantization.pt2e.quantizer import QuantizationAnnotation


def is_annotated(node: Node) -> bool:
    """Given a node return whether the node is annotated."""
    return (
        "quantization_annotation" in node.meta
        and cast(
            QuantizationAnnotation, node.meta["quantization_annotation"]
        )._annotated
    )


def is_output_annotated(node: Node) -> bool:
    """Given a node, return whether the output of the node is annotated."""
    if "quantization_annotation" in node.meta:
        annotation = cast(QuantizationAnnotation, node.meta["quantization_annotation"])
        return annotation._annotated and annotation.output_qspec is not None
    else:
        return False


def mark_node_as_annotated(node: Node) -> None:
    """Marks node as annotated. If needed, an empty  QuantizationAnnotation is added
    to the quantization_annotation node meta entry.
    """
    if "quantization_annotation" not in node.meta:
        node.meta["quantization_annotation"] = QuantizationAnnotation()
    node.meta["quantization_annotation"]._annotated = True


def is_ok_for_quantization(node: Node, gm: GraphModule):
    """Check if an node can be quantized. The node can not be quantized if:
    - The node does not output a float tensor or,
    - The node outputs a large scalar.
    """
    return not (is_non_float_tensor(node) or is_large_scalar(node, gm))


def get_node_target(module: torch.nn.Module | GraphModule, target_str: str):
    targets = target_str.split(".")
    for target in targets[:-1]:
        module = module.get_submodule(target)
    return getattr(module, targets[-1])


def is_large_scalar(node: Node, gm: GraphModule):
    """Check if input is a large scalar value. So that we can skip quantization for the node
    since histc op (in HistogramObserver) only works for values up to certain upper bound
    """
    if node.op == "get_attr" and isinstance(node.target, str):
        tensor = get_node_target(gm, node.target)
        # torch.histc works until this upper bound
        HISTC_UPPER_BOUND = 3.4028235e15
        return tensor.numel() == 1 and abs(tensor.item()) > HISTC_UPPER_BOUND
    return False


def is_non_float_tensor(node: Node) -> bool:
    """Check if the output of a node has a data type other than `torch.float32`.

    If the output is not `torch.float32`, quantization cannot be performed, as
    observers only work with floating-point tensors.

    Args:
        node (Node): The node to check the output(s) for.

    Returns:
        bool: `True` if the data type is not float32, otherwise `False`.

    Note:
        - If `node.meta["val"]` is a `list`, the function returns `True` if **any**
          element is **not** an instance of `FakeTensor` or does **not** have
          `torch.float32` as its data type.
        - If node.meta["val"] is missing or is not an instance of `FakeTensor`, the
          function returns True.
    """
    if "val" in node.meta and isinstance(node.meta["val"], Sequence):
        return any(
            not isinstance(fake_tensor, FakeTensor)
            or fake_tensor.dtype != torch.float32
            for fake_tensor in node.meta["val"]
        )

    if "val" not in node.meta or not isinstance(node.meta["val"], FakeTensor):
        return True

    return node.meta["val"].dtype != torch.float32
