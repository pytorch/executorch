# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class CanonicalizeGatherPass(ArmPass):
    """
    Canonicalize gather so it can be lowered to TOSA.GATHER via the backend dialect.

    This pass is intended to run only for nodes already gated by GatherSupported.

    Generic behavior:
    - Only insert bool<->int8 casts when x is bool:
        * If x is bool: gather runs on int8 and output is cast back to bool.
        * If x is not bool: gather runs on original dtype and output keeps dtype.

    2D behavior:
    - Reshape x from [N,K] to [N,K,1] so values matches TOSA gather's [N,K,C].
    - Reshape indices to [N,W].
    - Lower using tosa.GATHER.default, producing [N,W,1].
    - Reshape output to [N,W].

    3D behavior:
    - Permute and reshape x from [N,K,C] to [N*C,K,1].
    - Permute and reshape indices from [N,W,C] to [N*C,W].
    - Lower using tosa.GATHER.default, producing [N*C,W,1].
    - Reshape and permute output to [N,W,C].
    - Note that this decomposition requires the channel size of the indices to be the same as the channel
    size of the input, which is not guaranteed in Pytorch. We reject nodes not fulfilling this when partitioning.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    _TARGET_OPS = {exir_ops.edge.aten.gather.default}

    def call_operator(self, op, args, kwargs, meta):
        if op not in self._TARGET_OPS:
            return super().call_operator(op, args, kwargs, meta)

        # edge.aten.gather.default: (x, dim, index) with kw-only sparse_grad
        x, dim, index = args

        # GatherSupported should have gated this already; treat violations as errors.
        x_shape = x.data.shape
        index_shape = index.data.shape
        dim = dim % len(x_shape)
        if not (
            dim in (1,)
            and len(x_shape) in (2, 3)
            and len(index_shape) in (2, 3)
            and index_shape[0] == x_shape[0]
        ):
            raise RuntimeError(
                f"[{op}] Unexpected gather pattern; expected one of: "
                f"x:[N,K], index:[N,W], dim in {{1,}}, matching N, "
                f"x:[N,K,C], index:[N,W,C], dim in {{1,}}, matching N. "
                f"Got dim={dim}, x.shape={x_shape}, index.shape={index_shape}."
            )

        N, K = x_shape[0], x_shape[1]
        W = index_shape[1]

        view_op = exir_ops.edge.aten.view_copy.default
        to_copy_op = exir_ops.edge.dim_order_ops._to_dim_order_copy.default
        permute_copy_op = exir_ops.edge.aten.permute_copy.default

        # Use backend dialect gather:
        # values:  [N,K,C]
        # indices: [N,W]
        # output:  [N,W,C]
        tosa_gather_op = exir_ops.backend.tosa.GATHER.default

        needs_bool_cast = x.data.dtype == torch.bool

        # bool -> int8 (only if needed)
        values_in = x
        if needs_bool_cast:
            values_in = super().call_operator(
                to_copy_op,
                (x,),
                {"dtype": torch.int8},
                meta,
                updated=True,
            )

        # [N,K] -> [N,K,1]
        if len(x_shape) < 3:
            values_3d = super().call_operator(
                view_op,
                (values_in, [N, K, 1]),
                {},
                meta,
                updated=True,
            )

            # indices stays [N,W]
            gathered_3d = super().call_operator(
                tosa_gather_op,
                (values_3d, index),
                {},
                meta,
                updated=True,
            )
        else:
            # For 3D inputs we need to reshuffle both input and indices so that
            # the payload is accessed from the batch rather than the channels
            # since torch allows for accessing of different indices across channels
            # while TOSA does not.
            # 1. Permute input [N,K,C] to [N,C,K].
            # 2. Rehsape input [N,K,C] to [N*C,K,1].
            # 3. Permute indices [N,W,C] to [N,C,W].
            # 4. Reshape indices [N,C,W] to [N*C,W].
            # 5. Gather using tosa.GATHER with reshaped inputs and indices.
            # 6. Reshape output [N*C,W,1] to [N,C,W].
            # 7. Permute output [N,C,W] to [N,W,C].

            # TODO if index input is a repeat across last dim with C repeats
            # we can remove the repeat and directly lower to tosa.GATHER.
            values_3d = values_in
            N, K, C = x_shape
            _, W, _ = index_shape
            values_3d_permuted = super().call_operator(
                permute_copy_op,
                (values_3d, [0, 2, 1]),
                {},
                meta,
                updated=True,
            )  # [N,C,K]
            values_3d_reshaped = super().call_operator(
                view_op,
                (values_3d_permuted, [N * C, K, 1]),
                {},
                meta,
                updated=True,
            )  # [N*C,K,1]
            indices_permuted = super().call_operator(
                permute_copy_op,
                (index, [0, 2, 1]),
                {},
                meta,
                updated=True,
            )  # [N,L,W]
            indices_reshaped = super().call_operator(
                view_op,
                (indices_permuted, [N * C, W]),
                {},
                meta,
                updated=True,
            )  # [N*L,W]
            gathered = super().call_operator(
                tosa_gather_op,
                (values_3d_reshaped, indices_reshaped),
                {},
                meta,
                updated=True,
            )  # [N*L,W,1]
            gathered_reshaped = super().call_operator(
                view_op,
                (gathered, [N, C, W]),
                {},
                meta,
                updated=True,
            )  # [N,L,W]
            gathered_3d = super().call_operator(
                permute_copy_op,
                (gathered_reshaped, [0, 2, 1]),
                {},
                meta,
                updated=True,
            )  # [N,W,L]

        if len(x_shape) == 2:
            # [N,W,1] -> [N,W]
            gathered = super().call_operator(
                view_op,
                (gathered_3d, [N, W]),
                {},
                meta,
                updated=True,
            )
        else:
            gathered = gathered_3d

        # int8 -> bool (only if needed)
        if needs_bool_cast:
            return super().call_operator(
                to_copy_op,
                (gathered,),
                {"dtype": torch.bool},
                meta,
                updated=True,
            )

        return gathered
