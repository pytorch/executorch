# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import torch
from torch.fx import GraphModule
from torchao.quantization.pt2e.observer import HistogramObserver
from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
)
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY


@dataclass(eq=True, frozen=True)
class QuantizationConfig:
    input_activation: Optional[QuantizationSpec]
    output_activation: Optional[QuantizationSpec]


class AddQuantizer(Quantizer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _get_qspec():
        return QuantizationSpec(
            dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            qscheme=torch.per_tensor_symmetric,
            is_dynamic=False,
            observer_or_fake_quant_ctr=HistogramObserver.with_args(eps=2**-12),
        )

    @staticmethod
    def _get_qconfig():
        qspec = AddQuantizer._get_qspec()
        return QuantizationConfig(
            input_activation=qspec,
            output_activation=qspec,
        )

    def annotate(self, model: GraphModule):
        config = self._get_qconfig()
        annotated_partitions = []

        for node in model.graph.nodes:
            if node.op != "call_function" or node.target not in [
                torch.ops.aten.add.Tensor,
                torch.ops.aten.add_.Tensor,
            ]:
                continue

            if Q_ANNOTATION_KEY in node.meta and node.meta[Q_ANNOTATION_KEY]._annotated:
                continue

            input_qspec_map = {
                node.args[0]: config.input_activation,
                node.args[1]: config.input_activation,
            }

            node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=config.output_activation,
                _annotated=True,
            )
            annotated_partitions.append([node])

        return annotated_partitions

    def validate(self, model: GraphModule) -> None:
        pass


def check_count(
    graph_module: GraphModule, op: torch.fx.node.Target, expected_count: int
):
    actual_count = sum(
        1
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target == op
    )

    assert (
        actual_count == expected_count
    ), f"Expected {expected_count} {op} nodes, got {actual_count}"


def get_node_args(graph_module: GraphModule, op: torch.fx.node.Target):
    """Helper to get arguments of specific operator nodes"""
    nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target == op
    ]
    return [node.args for node in nodes]
