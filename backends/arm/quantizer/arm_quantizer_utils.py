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

from typing import cast

from torch.fx import Node

from torchao.quantization.pt2e.quantizer import QuantizationAnnotation
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY


def is_annotated(node: Node) -> bool:
    """Given a node return whether the node is annotated."""
    return (
        Q_ANNOTATION_KEY in node.meta
        and cast(QuantizationAnnotation, node.meta[Q_ANNOTATION_KEY])._annotated
    )


def is_output_annotated(node: Node) -> bool:
    """Given a node, return whether the output of the node is annotated."""
    if Q_ANNOTATION_KEY in node.meta:
        annotation = cast(QuantizationAnnotation, node.meta[Q_ANNOTATION_KEY])
        return annotation._annotated and annotation.output_qspec is not None
    else:
        return False


def mark_node_as_annotated(node: Node) -> None:
    """Marks node as annotated. If needed, an empty  QuantizationAnnotation is added
    to the quantization_annotation node meta entry.
    """
    if Q_ANNOTATION_KEY not in node.meta:
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation()
    node.meta[Q_ANNOTATION_KEY]._annotated = True
