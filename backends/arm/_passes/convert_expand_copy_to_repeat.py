# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging
from typing import cast, Set, Type

import torch

from executorch.backends.arm._passes.unsqueeze_before_repeat_pass import (
    UnsqueezeBeforeRepeatPass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

logger = logging.getLogger(__name__)


def calculate_multiples(args):
    input_node_or_tensor = args[0]

    if isinstance(input_node_or_tensor, torch.fx.node.Node):
        input_data = input_node_or_tensor.meta["val"]
    else:
        input_data = input_node_or_tensor.data

    input_shape = input_data.shape

    multiples = cast(list[int], args[1])
    expanded_rank = len(multiples)

    # Expanded shape is 'input_shape' front-padded with ones.
    padding = expanded_rank - len(input_shape)
    extended_shape = [
        input_shape[i] if i >= 0 else 1 for i in range(-padding, len(input_shape))
    ]

    # To convert expand arg to repeat arg, non-repeated dims should have
    # multiples[dim] = 1. Passing -1 to expand arg means
    # not changing the size of that dimension.
    multiples = [
        multiples[i] if multiples[i] != -1 and extended_shape[i] == 1 else 1
        for i in range(expanded_rank)
    ]
    return multiples


class ConvertExpandCopyToRepeatPass(ExportPass):
    """
    Replace expand copy with repeat since it is a repeat that can only repeat singleton dimensions.
    """

    _passes_required_after: Set[Type[ExportPass]] = {UnsqueezeBeforeRepeatPass}

    expand_copy = exir_ops.edge.aten.expand_copy.default
    repeat = exir_ops.edge.aten.repeat.default

    def call_operator(self, op, args, kwargs, meta):
        if op != self.expand_copy:
            return super().call_operator(op, args, kwargs, meta)

        multiples = calculate_multiples(args)

        if all((x == 1 for x in multiples)):
            # All dimensions/repetitions occur only once. Remove node
            # altogether since it's in practice just a copy.
            logger.warning("Found redundant expand node (no-op). Removing it.")

            return args[0]

        return super().call_operator(
            op=self.repeat, args=(args[0], multiples), kwargs=kwargs, meta=meta
        )
