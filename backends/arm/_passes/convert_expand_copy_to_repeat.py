# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import cast

from executorch.backends.arm.tosa_mapping import extract_tensor_meta
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class ConvertExpandCopyToRepeatPass(ExportPass):
    """
    Replace expand copy with repeat since it is a repeat that can only repeat singleton dimensions.
    """

    expand_copy = exir_ops.edge.aten.expand_copy.default
    repeat = exir_ops.edge.aten.repeat.default

    def call_operator(self, op, args, kwargs, meta):
        if op != self.expand_copy:
            return super().call_operator(op, args, kwargs, meta)

        _, shape, _ = extract_tensor_meta(meta.data)
        multiples = cast(list[int], args[1])
        expanded_rank = len(multiples)

        # Expanded shape is 'shape' front-padded with ones.
        padding = expanded_rank - len(shape)
        extended_shape = [
            shape[i] if i >= 0 else 1 for i in range(-padding, len(shape))
        ]

        # To convert expand arg to repeat arg, non-repeated dims should have
        # multiples[dim] = 1. Passing -1 to expand arg means
        # not changing the size of that dimension.
        multiples = [
            multiples[i] if multiples[i] != -1 and extended_shape[i] == 1 else 1
            for i in range(expanded_rank)
        ]
        return super().call_operator(
            op=self.repeat, args=(args[0], multiples), kwargs=kwargs, meta=meta
        )
