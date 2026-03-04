# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Set, Type

import torch
from executorch.backends.arm._passes.get_decomposition_pass import GetDecompositionPass
from executorch.exir.pass_base import ExportPass


class DecomposeMatmulPass(GetDecompositionPass):
    """Decompose aten.matmul into more primitive ops using PyTorch decomposition table.
    By defualt, to_edge will decompose torch.matmul into 1d (dot), 2d (mm), 3d (bmm), ops
    along with required reshape operators and expands/repeats. For a quanitzed matmul this
    is problmeatic as the mm/bmm nodes will not be surrounded by quant-dequant nodes which
    makes it hard for the backend to identify them as quantized matmuls. For instance:
        q -> dq -> matmul -> q -> dq
    would be decomposed to:
        q -> dq -> expand -> reshape -> mm/bmm -> reshape -> q -> dq

    By decomposing matmul early, we can ensure that the quant-dequant nodes surround all
    the nodes of the decomposition. For instance:
        q -> dq -> matmul -> q -> dq
    would be decomposed to:
        q -> dq -> expand -> q -> dq -> reshape -> q -> dq -> mm/bmm -> q -> dq -> reshape -> q -> dq
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    targeted_ops = [
        torch.ops.aten.matmul.default,
    ]

    def _skip_pass(self, input_tensors: list):
        # TODO Add support for mutliplication with vectors
        if len(input_tensors) > 1:
            if input_tensors[1].dim() == 1:
                return True
        else:
            if input_tensors[0].dim() == 1:
                return True

        return False
