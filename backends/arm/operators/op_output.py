# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import cast

import serializer.tosa_serializer as ts
import torch


def process_output(
    node: torch.fx.Node,
    tosa_graph: ts.TosaSerializer,
):
    for output in cast(tuple[torch.fx.Node, ...], node.args[0]):
        tosa_graph.addOutputTensor(
            tosa_graph.currRegion.currBasicBlock.tensors[output.name]
        )
