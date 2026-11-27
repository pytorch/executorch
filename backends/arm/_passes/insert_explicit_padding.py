# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, ProxyValue


class InsertExplicitPaddingPass(ArmOpTargetedPass):
    """Materialize dynamic convolution and pooling padding as explicit PAD ops.

    Static large-stride max pooling can also use explicit padding to avoid
    incorrect padded pooling results from the U85 compiler path.

    Args:
        large_stride_pool_padding (bool): Materialize static max-pool padding
            when either stride exceeds three. Defaults to False.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()
    target_ops = (
        exir_ops.backend.tosa.CONV2D.default,
        exir_ops.backend.tosa.CONV3D.default,
        exir_ops.backend.tosa.DEPTHWISE_CONV2D.default,
        exir_ops.backend.tosa.MAX_POOL2D.default,
        exir_ops.backend.tosa.AVG_POOL2D.default,
    )

    def __init__(self, large_stride_pool_padding: bool = False):
        super().__init__()
        self.large_stride_pool_padding = large_stride_pool_padding

    def _is_dynamic_padding(
        self, padding: list[int | ProxyValue] | tuple[int | ProxyValue, ...]
    ) -> bool:
        return isinstance(padding, (list, tuple)) and any(
            isinstance(p, ProxyValue) for p in padding
        )

    def call_operator(self, op, args, kwargs, meta, updated=False) -> ProxyValue:
        if op not in self.target_ops:
            return super().call_operator(op, args, kwargs, meta, updated)
        if op == exir_ops.backend.tosa.MAX_POOL2D.default:
            padding_index = 3
        elif op == exir_ops.backend.tosa.AVG_POOL2D.default:
            padding_index = 5
        else:
            padding_index = 4
        padding = args[padding_index]
        static_pool_padding = (
            self.large_stride_pool_padding
            and op == exir_ops.backend.tosa.MAX_POOL2D.default
            and max(args[2]) > 3
            and not self._is_dynamic_padding(padding)
            and any(padding)
        )
        if not self._is_dynamic_padding(padding) and not static_pool_padding:
            return super().call_operator(op, args, kwargs, meta, updated)

        # Create a pad op before the convolution/pool op.
        input_tensor = args[0]

        if static_pool_padding and any(p < 0 for p in padding):
            # Adjusted pool padding can crop unused input edges. TOSA PAD
            # cannot crop; clamping alone would change the pooling extent.
            # Emit the crop here, after the earlier ATen pad/slice rewrites.
            starts = [0] * 4
            sizes = list(input_tensor.data.shape)
            for axis in range(2):
                before, after = padding[axis * 2 : axis * 2 + 2]
                starts[axis + 1] = max(-before, 0)
                sizes[axis + 1] += min(before, 0) + min(after, 0)
            start_shape = super().call_shape_operator(
                exir_ops.backend.tosa.CONST_SHAPE.default, (starts,), {}, meta, True
            )
            size_shape = super().call_shape_operator(
                exir_ops.backend.tosa.CONST_SHAPE.default, (sizes,), {}, meta, True
            )
            input_tensor = super().call_operator(
                exir_ops.backend.tosa.SLICE.default,
                (input_tensor, start_shape, size_shape),
                {},
                meta,
                True,
            )
            padding = [max(p, 0) for p in padding]

        zero_padding_pair = [0, 0]
        spatial_rank = 3 if op == exir_ops.backend.tosa.CONV3D.default else 2
        zero_spatial_padding = [0] * (spatial_rank * 2)
        padding_shape: list[int | ProxyValue] | ProxyValue = [
            *zero_padding_pair,
            *padding,
            *zero_padding_pair,
        ]
        if static_pool_padding:
            padding_shape = super().call_shape_operator(
                exir_ops.backend.tosa.CONST_SHAPE.default,
                (padding_shape,),
                {},
                meta,
                True,
            )

        pad_value = args[6] if len(args) == 8 else None
        if pad_value is None:
            value = (
                torch.iinfo(input_tensor.data.dtype).min if static_pool_padding else 0
            )
            pad_value = super().call_operator(
                exir_ops.backend.tosa.CONST.default,
                ([value], [1], input_tensor.data.dtype),
                {},
                meta,
                True,
            )
        pad_res = super().call_operator(
            exir_ops.backend.tosa.PAD.default,
            (
                input_tensor,
                padding_shape,
                pad_value,
            ),
            {},
            meta,
            True,
        )
        new_args = list(args)
        new_args[0] = pad_res
        new_args[padding_index] = zero_spatial_padding
        return super().call_operator(op, tuple(new_args), kwargs, meta, updated)
