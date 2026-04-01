# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from typing import Set, Type

import torch

from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class RewritePadPass(ArmPass):
    """Rewrite constant_pad_nd operator to TOSA Pad operator with constant
    mode.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()
    targeted_ops = {
        exir_ops.edge.aten.constant_pad_nd.default,
        exir_ops.edge.aten.pad.default,
    }

    def _rewrite_constant_pad(self, input_tensor, pad, value, meta):
        output_dtype = meta["val"].dtype
        if output_dtype in (torch.int8, torch.int16):
            input_qparams = meta.data.get("input_qparams", {})
            if len(input_qparams) == 0:
                raise ValueError(
                    f"No input quantization parameters found in metadata for constant_pad_nd with output dtype {output_dtype}"
                )
            value = input_qparams[0].quantize_value(value).item()

        # Each dim needs 2 padding values. For example, to pad the last dimension, the pad has the form
        # (padding_left, padding_right); to pad the last two dimensions, the pad has the form
        # (padding_left, padding_right, padding_top, padding_bottom), and so on. We want to reverse the padding
        # so that we get (N_before, N_after, C_before, C_after, H_before, H_after, W_before, W_after) for a 4D
        # input tensor.
        pad_pairs = [[pad[i], pad[i + 1]] for i in range(0, len(pad), 2)]
        input_pad = []
        for pair in reversed(pad_pairs):
            input_pad.extend(pair)
        input_rank = len(input_tensor.data.shape)
        # Place spatial dimensions last and pad non-spatial dimensions with 0 padding
        shape = [0] * ((input_rank * 2 - len(pad))) + input_pad

        pad_shape = super().call_shape_operator(
            exir_ops.backend.tosa.CONST_SHAPE.default, (shape,), {}, meta, True
        )

        return super().call_operator(
            exir_ops.backend.tosa.PAD.default,
            (input_tensor, pad_shape),
            {"value": value},
            meta,
            True,
        )

    def _slice_idx(self, x, dim: int, idx: int, meta):
        return super().call_operator(
            exir_ops.edge.aten.slice_copy.Tensor,
            (x, dim, idx, idx + 1),
            {},
            meta,
            True,
        )

    def _pad_along_dim(
        self,
        x,
        dim: int,
        left: int,
        right: int,
        mode: str,
        meta,
    ):
        if left == 0 and right == 0:
            return x

        size = x.data.shape[dim]
        if isinstance(size, torch.SymInt):
            raise ValueError(f"Pad mode '{mode}' does not support symbolic shape yet.")
        if not isinstance(size, int):
            raise ValueError(f"Expected integer dim size for pad rewrite, got {size}.")

        left_tensors = []
        right_tensors = []

        if mode == "replicate":
            left_tensors = [self._slice_idx(x, dim, 0, meta) for _ in range(left)]
            right_tensors = [
                self._slice_idx(x, dim, size - 1, meta) for _ in range(right)
            ]
        elif mode == "circular":
            left_tensors = [
                self._slice_idx(x, dim, size - left + i, meta) for i in range(left)
            ]
            right_tensors = [self._slice_idx(x, dim, i, meta) for i in range(right)]
        elif mode == "reflect":
            if left >= size or right >= size:
                raise ValueError(
                    f"Pad mode 'reflect' requires pad < input size, got left={left}, right={right}, size={size}."
                )
            left_tensors = [
                self._slice_idx(x, dim, left - i, meta) for i in range(left)
            ]
            right_tensors = [
                self._slice_idx(x, dim, size - 2 - i, meta) for i in range(right)
            ]
        else:
            raise ValueError(f"Unsupported pad mode '{mode}'.")

        return super().call_operator(
            exir_ops.edge.aten.cat.default,
            (left_tensors + [x] + right_tensors, dim),
            {},
            meta,
            True,
        )

    def _rewrite_non_constant_pad(
        self,
        input_tensor,
        pad: Sequence[int],
        mode: str,
        meta,
    ):
        if len(pad) % 2 != 0:
            raise ValueError(f"Invalid pad spec length {len(pad)} for mode '{mode}'.")

        output = input_tensor
        pairs = [(pad[i], pad[i + 1]) for i in range(0, len(pad), 2)]
        rank = len(input_tensor.data.shape)
        for pair_idx, (left, right) in enumerate(pairs):
            if not isinstance(left, int) or not isinstance(right, int):
                raise ValueError(
                    f"Pad mode '{mode}' expects integer pad values, got ({left}, {right})."
                )
            # F.pad pad tuples are ordered from the innermost dimension outward.
            dim = rank - 1 - pair_idx
            output = self._pad_along_dim(output, dim, left, right, mode, meta)
        return output

    def call_operator(self, op, args, kwargs, meta, updated=False):
        if op not in self.targeted_ops:
            return super().call_operator(op, args, kwargs, meta)

        if op == exir_ops.edge.aten.constant_pad_nd.default:
            if len(args) == 3:
                input_tensor, pad, value = args
            else:
                input_tensor, pad = args
                value = 0
            return self._rewrite_constant_pad(input_tensor, pad, value, meta)

        if len(args) < 2:
            raise ValueError(
                f"Expected at least 2 args for aten.pad.default, got {args}"
            )

        input_tensor, pad = args[:2]
        mode = args[2] if len(args) > 2 else kwargs.get("mode", "constant")
        value = args[3] if len(args) > 3 else kwargs.get("value", 0)

        if not isinstance(mode, str):
            raise ValueError(f"Expected string mode in aten.pad.default, got {mode}")

        if mode == "constant":
            return self._rewrite_constant_pad(input_tensor, pad, value, meta)

        if mode in ("reflect", "replicate", "circular"):
            return self._rewrite_non_constant_pad(input_tensor, pad, mode, meta)

        raise ValueError(f"Unsupported pad mode '{mode}' in aten.pad.default.")
