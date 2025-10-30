# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchao.quantization.pt2e.quantizer import QuantizationAnnotation
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY


def _nodes_are_annotated(node_list):
    for node in node_list:
        quantization_annotation = node.meta.get(Q_ANNOTATION_KEY, None)
        if not quantization_annotation:
            return False
        if quantization_annotation._annotated:
            continue
        else:
            return False
    return True


def _annotate_nodes(node_tuples, quant_spec, input_node=False):
    for node_tuple in node_tuples:
        node = node_tuple[0]
        quant_annotation = node.meta.get(
            Q_ANNOTATION_KEY, QuantizationAnnotation(_annotated=True)
        )
        if input_node:
            input_node = node_tuple[1]
            quant_annotation.input_qspec_map[input_node] = quant_spec
        else:
            quant_annotation.output_qspec = quant_spec
        node.meta[Q_ANNOTATION_KEY] = quant_annotation
