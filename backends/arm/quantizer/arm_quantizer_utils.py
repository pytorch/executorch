# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Provide utilities for quantization annotations.

Use these helpers to check and mark annotation state when working with
``QuantizationAnnotation`` entries in FX node metadata.

"""

from typing import cast

from executorch.backends.arm.common.annotation_meta import ArmAnnotationInfo

from torch.fx import Node

from torchao.quantization.pt2e.quantizer import QuantizationAnnotation
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY


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
    meta_custom[ArmAnnotationInfo.CUSTOM_META_KEY] = annotation_info
    node.meta["custom"] = meta_custom
