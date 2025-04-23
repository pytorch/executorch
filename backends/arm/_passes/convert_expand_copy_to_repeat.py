# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging
from typing import cast

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

logger = logging.getLogger(__name__)


class ConvertExpandCopyToRepeatPass(ExportPass):
    """
    Replace expand copy with repeat since it is a repeat that can only repeat singleton dimensions.
    """

    expand_copy = exir_ops.edge.aten.expand_copy.default
    repeat = exir_ops.edge.aten.repeat.default

    def call_operator(self, op, args, kwargs, meta):
        if op != self.expand_copy:
            return super().call_operator(op, args, kwargs, meta)

        input_shape = args[0].data.shape
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

        if all((x == 1 for x in multiples)):
            # All dimensions/repetitions occur only once. Remove node
            # altogether since it's in practice just a copy.
            logger.warning("Found redundant expand node (no-op). Removing it.")

            return args[0]

        return super().call_operator(
            op=self.repeat, args=(args[0], multiples), kwargs=kwargs, meta=meta
        )
