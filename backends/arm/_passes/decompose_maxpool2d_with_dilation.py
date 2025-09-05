# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import operator

from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops

# We'll decompose only the EXIR edge max_pool2d ops when dilation > 1
EDGE_MAXPOOL2D = (
    exir_ops.edge.aten.max_pool2d.default,
    exir_ops.edge.aten.max_pool2d_with_indices.default,
)


class DecomposeMaxPool2DPass(ArmPass):
    """
    Decompose dilated max_pool2d (EXIR edge ops) into space-to-batch -> maxpool -> batch-to-space.
    """

    def call_operator(self, op, args, kwargs, meta):
        # Only intercept EXIR edge max_pool2d ops
        if op not in EDGE_MAXPOOL2D:
            return super().call_operator(op, args, kwargs, meta)

        # detect whether indices variant
        is_with_indices = op is exir_ops.edge.aten.max_pool2d_with_indices.default

        # Normalize missing trailing args to their defaults
        x = args[0]
        kernel_size = args[1]
        stride = args[2]
        padding = args[3] if len(args) >= 4 else 0
        dilation = args[4] if len(args) >= 5 else 1
        ceil_mode = args[5] if len(args) == 6 else False

        # Normalize attributes
        pad_h, pad_w = (padding, padding) if isinstance(padding, int) else padding
        d_h, d_w = (dilation, dilation) if isinstance(dilation, int) else dilation
        k_h, k_w = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        s_h, s_w = (stride, stride) if isinstance(stride, int) else stride

        # If no dilation: call EXIR edge op
        if d_h == 1 and d_w == 1:
            minimal_args = [x, kernel_size, stride, padding, dilation, ceil_mode]
            return super().call_operator(op, tuple(minimal_args), {}, meta)

        # Compute padded and packed dimensions for dilation > 1
        N, C, H, W = x.data.size()
        ph, pw = pad_h, pad_w
        ph2, pw2 = pad_h, pad_w
        H_pad = H + ph + ph2
        W_pad = W + pw + pw2
        H_pack = (H_pad + d_h - 1) // d_h
        W_pack = (W_pad + d_w - 1) // d_w
        extra_h = 0 if H_pack < k_h else (s_h - ((H_pack - k_h) % s_h)) % s_h
        extra_w = 0 if W_pack < k_w else (s_w - ((W_pack - k_w) % s_w)) % s_w
        ph2 += extra_h * d_h
        pw2 += extra_w * d_w

        # 1) Pad via EXIR edge pad (preserves dtype)
        pad_edge = exir_ops.edge.aten.constant_pad_nd.default
        pads = [pw, pw2, ph, ph2, 0, 0, 0, 0]
        x_pad = super().call_operator(
            pad_edge,
            (x, pads, 0),
            {},
            meta,
        )

        # 2) Space-to-batch: reshape and permute
        x2 = super().call_operator(
            exir_ops.edge.aten.view_copy.default,
            (x_pad, [N, C, H_pack, d_h, W_pack, d_w]),
            {},
            meta,
        )
        x2 = super().call_operator(
            exir_ops.edge.aten.permute_copy.default,
            (x2, [3, 5, 0, 1, 2, 4]),
            {},
            meta,
        )
        x2 = super().call_operator(
            exir_ops.edge.aten.view_copy.default,
            (x2, [N * d_h * d_w, C, H_pack, W_pack]),
            {},
            meta,
        )

        # 3) Core pooling on packed tensor
        pool_edge_op = (
            exir_ops.edge.aten.max_pool2d_with_indices.default
            if is_with_indices
            else exir_ops.edge.aten.max_pool2d.default
        )
        pool_args = (x2, (k_h, k_w), (s_h, s_w), (0, 0), 1, ceil_mode)
        pool_out = super().call_operator(
            pool_edge_op,
            pool_args,
            {},
            meta,
        )

        # Unpack pooled result
        if is_with_indices:
            pooled_proxy = super().call_operator(
                operator.getitem,
                (pool_out, 0),
                {},
                meta,
            )
            indices_proxy = super().call_operator(
                operator.getitem,
                (pool_out, 1),
                {},
                meta,
            )
            pooled_fake, _ = pool_out.data
        else:
            pooled_proxy = pool_out
            pooled_fake = pool_out.data
            indices_proxy = None

        _, C_out, H_out, W_out = pooled_fake.shape

        # 4) Batch-to-space: reshape and permute back
        out = super().call_operator(
            exir_ops.edge.aten.view_copy.default,
            (pooled_proxy, [d_h, d_w, N, C_out, H_out, W_out]),
            {},
            meta,
        )
        out = super().call_operator(
            exir_ops.edge.aten.permute_copy.default,
            (out, [2, 3, 4, 0, 5, 1]),
            {},
            meta,
        )
        # now flatten back into (N, C, H_out*d_h, W_out*d_w)
        out = super().call_operator(
            exir_ops.edge.aten.view_copy.default,
            (out, [N, C_out, H_out * d_h, W_out * d_w]),
            {},
            meta,
        )

        # 5) Final crop
        S_top = ph // d_h + (1 if ph % d_h else 0)
        S_left = pw // d_w + (1 if pw % d_w else 0)
        S_top = max(0, min(S_top, H_out * d_h - H))
        S_left = max(0, min(S_left, W_out * d_w - W))
        out = super().call_operator(
            exir_ops.edge.aten.slice_copy.Tensor,
            (out, 2, S_top, S_top + H),
            {},
            meta,
        )
        out = super().call_operator(
            exir_ops.edge.aten.slice_copy.Tensor,
            (out, 3, S_left, S_left + W),
            {},
            meta,
        )

        if is_with_indices:
            # Reconstruct indices
            idx = super().call_operator(
                exir_ops.edge.aten.view_copy.default,
                (indices_proxy, [d_h, d_w, N, C_out, H_out, W_out]),
                {},
                meta,
            )
            idx = super().call_operator(
                exir_ops.edge.aten.permute_copy.default,
                (idx, [2, 3, 4, 0, 5, 1]),
                {},
                meta,
            )
            idx = super().call_operator(
                exir_ops.edge.aten.view_copy.default,
                (idx, [N, C_out, H_out * d_h, W_out * d_w]),
                {},
                meta,
            )
            idx = super().call_operator(
                exir_ops.edge.aten.slice_copy.Tensor,
                (idx, 2, S_top, S_top + H),
                {},
                meta,
            )
            idx = super().call_operator(
                exir_ops.edge.aten.slice_copy.Tensor,
                (idx, 3, S_left, S_left + W),
                {},
                meta,
            )
            return out, idx

        return out
