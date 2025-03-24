# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.library import impl, Library

op_lib = Library("qaisw", "DEF")
op_lib.define("embedding(Tensor table, Tensor indices) -> Tensor")

@impl(op_lib, "embedding", dispatch_key="CompositeExplicitAutograd")
def embedding_impl(table: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return table[indices]


class CustomEmbedding(torch.nn.Module):
    def __init__(self, weight):
        super(CustomEmbedding, self).__init__()
        self.weight = weight

    def forward(self, indices):
        return torch.ops.qaisw.embedding.default(self.weight, indices)


def custom_embedding_annotation(gm: torch.fx.GraphModule) -> None:
    import itertools
    from executorch.backends.qualcomm.quantizer.annotators import (
        _is_annotated,
        QUANT_ANNOTATION_KEY,
    )
    from executorch.backends.qualcomm.quantizer.qconfig import (
        get_16a4w_qnn_ptq_config,
    )
    from torch.ao.quantization.quantize_pt2e import QuantizationAnnotation, SharedQuantizationSpec
    from torch.fx import Node
    from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

    custom_partitions = get_source_partitions(gm.graph, [torch.ops.qaisw.embedding.default])
    custom_partitions = list(itertools.chain(*custom_partitions.values()))
    quantization_config = get_16a4w_qnn_ptq_config()
    for custom_partition in custom_partitions:
        if len(custom_partition.output_nodes) > 1:
            raise ValueError("custom partition has more than one output node")
        custom_node = custom_partition.output_nodes[0]
        if (
            custom_node.op != "call_function"
            or custom_node.target != torch.ops.qaisw.embedding.default
        ):
            raise ValueError(f"{custom_node} is not a custom operator")
        # skip annotation if it is already annotated
        if _is_annotated([custom_node]):
            continue

        input_qspec_map = {}
        input_act = custom_node.args[0]
        assert isinstance(input_act, Node)
        input_spec = quantization_config.weight
        input_qspec_map[input_act] = input_spec

        custom_node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=SharedQuantizationSpec((input_act, custom_node)),
            _annotated=True,
        )
