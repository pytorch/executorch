# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared-qspec annotation for cat under QAT.

The per-node ``Cat`` annotators put a ``ConcatObserver`` on the output, which
aligns the input ranges by writing its ``min_val`` / ``max_val`` onto the input
observers. That only works for PTQ observers. Under QAT:

- the output gets a plain observer, so the cat output is never fake-quantized;
- the inputs are ``FakeQuantize`` modules that keep their range on
  ``activation_post_process``, so the write lands on an unused attribute and
  every input trains with its own scale.

One ``SharedQuantizationSpec`` across all inputs and the output gives the whole
cat a single ``FakeQuantize``, as the XNNPACK quantizer and BoltNN do.
"""

from typing import Callable, Optional, Set

import torch
from torch.fx import GraphModule, Node
from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    SharedQuantizationSpec,
)

from .qconfig import QuantizationConfig
from .rules import _is_annotated, _is_float_tensor, Q_ANNOTATION_KEY

CAT_TARGETS = (torch.ops.aten.cat.default, torch.ops.aten.concat.default)


def annotate_cat_shared_qspec(
    gm: GraphModule,
    get_quant_config: Callable[[Node], Optional[QuantizationConfig]],
    discard_nodes: Set[str],
) -> int:
    """Annotate every float cat with one qspec shared by its inputs and output.

    Must run before the per-node annotation pass, which then skips the claimed
    nodes. Returns the number of cats annotated.
    """
    count = 0
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in CAT_TARGETS:
            continue
        if node.name in discard_nodes or _is_annotated([node]):
            continue
        if not _is_float_tensor(node):
            continue
        quantization_config = get_quant_config(node)
        if quantization_config is None or quantization_config.input_activation is None:
            continue

        inputs = node.args[0]
        shared_qspec = SharedQuantizationSpec((inputs[0], node))
        input_qspec_map = {input_node: shared_qspec for input_node in inputs[1:]}
        input_qspec_map[inputs[0]] = quantization_config.input_activation
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=shared_qspec,
            _annotated=True,
        )
        count += 1
    return count
