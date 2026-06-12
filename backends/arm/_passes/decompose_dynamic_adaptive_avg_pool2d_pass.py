# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.arm._passes.decompose_adaptive_avg_pool2d_pass import (
    _get_decomposition,
    aten_ops,
    DecomposeAdaptiveAvgPool2dPass,
    edge_ops,
)
from executorch.backends.arm._passes.rewrite_adaptive_avg_pool2d import (
    RewriteAdaptiveAvgPool2dPass,
)


class DecomposeDynamicAdaptiveAvgPool2dPass(DecomposeAdaptiveAvgPool2dPass):
    """Decompose symbolic irregular AdaptiveAvgPool2d to TOSA shape ops.

    Directly representable dynamic cases are left to
    ``RewriteAdaptiveAvgPool2dPass``. Static cases stay in
    ``DecomposeAdaptiveAvgPool2dPass``.

    """

    _passes_required_after = {RewriteAdaptiveAvgPool2dPass}

    def call_operator(self, op, args, kwargs, meta, updated=False):
        if op not in (edge_ops + aten_ops) or not self.allowed_to_transform(meta):
            return super().call_operator(op, args, kwargs, meta, updated)

        x = args[0]
        output_size_h, output_size_w = args[1]
        if isinstance(output_size_h, torch.SymInt) or isinstance(
            output_size_w, torch.SymInt
        ):
            return super().call_operator(op, args, kwargs, meta, updated)

        if not self._has_dynamic_spatial_shape(x):
            return super().call_operator(op, args, kwargs, meta, updated)

        if self._is_dynamic_direct_case(x, output_size_h, output_size_w):
            return super().call_operator(op, args, kwargs, meta, updated)

        if not self._supports_dynamic_tosa_adaptive():
            return super().call_operator(op, args, kwargs, meta, updated)

        _, _, input_size_h, input_size_w = x.data.shape
        if self._is_static_shape(input_size_h, input_size_w):
            return super().call_operator(op, args, kwargs, meta, updated)

        _, _, cat_op = _get_decomposition(op)
        return self._decompose_dynamic_static_output(
            x, cat_op, output_size_h, output_size_w, kwargs, meta
        )
