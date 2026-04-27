# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from itertools import zip_longest
from typing import Sequence, Set, Type

import torch

from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import meta_without_qparams
from executorch.backends.arm._passes.convert_expand_copy_to_repeat import (
    ConvertExpandCopyToRepeatPass,
)
from executorch.backends.arm._passes.convert_squeezes_to_view import (
    ConvertSqueezesToViewPass,
)
from executorch.backends.arm._passes.replace_scalar_with_tensor_pass import (
    ReplaceScalarWithTensorByProfilePass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


def get_index_tensor_decomposition(op):
    """Return the operator overloads used to lower index.Tensor via TOSA gather.

    Raises:
        RuntimeError: If the provided operator is not supported by this pass.

    """
    if op is exir_ops.edge.aten.index.Tensor:
        return (
            exir_ops.edge.aten.view_copy.default,
            exir_ops.edge.aten.unsqueeze_copy.default,
            exir_ops.edge.aten.expand_copy.default,
            exir_ops.edge.aten.mul.Scalar,
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
            exir_ops.backend.tosa.GATHER.default,
        )

    raise RuntimeError(f"Can't get index.Tensor decomposition for op {op}")


def _broadcast_shape(
    shapes: Sequence[Sequence[int]],
) -> list[int]:
    """Compute the broadcasted shape (PyTorch/Numpy semantics) for a list of
    shapes.

    Requirements:
      - static shape only
      - shapes are right-aligned; lower-rank shapes are implicitly front-padded with 1s
      - per-axis dims must either match exactly or be 1

    Raises:
        RuntimeError: If shapes are not broadcastable.

    """
    out: list[int] = []
    # Reverse shapes to iterate trailing dims first (right-aligned); pad missing leading dims with 1.
    for axis, dims in enumerate(
        zip_longest(*(reversed(s) for s in shapes), fillvalue=1)
    ):
        chosen = max(dims)
        if any((d != 1 and d != chosen) for d in dims):
            raise RuntimeError(
                f"non-broadcastable dims at axis -{axis+1}: {list(dims)}"
            )
        out.insert(0, chosen)

    return out


class DecomposeIndexTensorToGatherPass(ArmPass):
    """Decompose edge.aten.index.Tensor into backend TOSA gather (+ basic
    arith).

    Supported subset:
      y = x.index([i0, i1, ..., i{m-1}])

    where each ik is a Tensor index, and m is the number of index tensors.

    Constraints:
      - `indices` list contains only Tensor indices (no None/slice/ellipsis)
      - Each index tensor dtype is int32
      - Index tensor shapes are broadcastable to a common shape `S` (per index.Tensor semantics)
      - Only prefix indexing is supported: the `m` tensor indices select elements
            from the first `m` dimensions of `x`, so `m <= rank(x)`.
      - Static shapes are required
      - If `x` has more than 2^31 elements, the computed linear index may overflow int32.

    Lowering strategy (single gather)
    ---------------------------------
    Let:
      - `S` be the broadcasted index shape
      - `W = prod(S)` (number of indexed positions)
      - `K = prod(x.shape[:m])` (flattened size of the indexed prefix)
      - `C = prod(x.shape[m:])` (flattened size of the trailing slice per index)
      - `trailing = x.shape[m:]`

    Steps:
    1) Compute parameters needed to lower index.Tensor
         - `S`, `W`, `K`, `C`, `trailing`
         - `lin_scales[i] = stride_i // C`, where `stride_i` are the contiguous-style
           strides derived from `x.shape` (for dim i).
    2) Reshape x to `[1, K, C]` (`x_1kc`).
    3) Build linear indices (`lin_1w`) by scaling each flattened index and summing:
         lin_1w = unsqueeze0( sum_{i=0..m-1} ( idx_flat[i] * lin_scales[i] ) )
       where:
         - `m = len(indices)`
         - `idx_flat[i]` is the i-th index tensor after broadcast to `S` and flatten to `[W]`
         - `lin_1w` has shape `[1, W]` and is used as the `indices` input to `tosa.GATHER`
    4) Single gather:
         `tosa.GATHER(x=x_1kc, indices=lin_1w) -> [1,W,C]`
    5) Reshape result to `[*S, *trailing]`.

    Example
    -------
    Consider:
        x.shape = [2, 3, 4]
        indices = [i0, i1]   # m = 2
        i0.shape = [2, 1]
        i1.shape = [1, 2]

    1) The index shapes broadcast to:
        S := [2, 2]
        W := prod(S) = 4

    We index the first m=2 dimensions of x, so:
        K := prod(x.shape[:m]) = 2 * 3 = 6
        C := prod(x.shape[m:]) = 4
        trailing := x.shape[m:] = [4]

    Contiguous strides of x are [12, 4, 1], so:
        lin_scales := [stride0 // C, stride1 // C] = [12//4, 4//4] = [3, 1]

    2) Values are reshaped to:
        x_1kc = view(x, [1, K, C]) = [1, 6, 4]

    3) After broadcasting and flattening the indices to length W:
        i0_broadcast, i1_broadcast have shape S=[2,2]
        i0_flat, i1_flat have shape [W]=[4]

    Linear indices are computed as:
        lin_w
            = lin_scales * [i0_flat, i1_flat]
            = 3 * i0_flat + 1 * i1_flat          # shape [W]
        lin_w is reshaped to [1, W] to match tosa.Gather semantics

    4) Single Gather:
        out_1wc = tosa.GATHER(values=x_1kc, indices=lin_1w)  # [1, 4, 4]

    5) Reshape result:
        out = view(out_1wc, [*S, *x.shape[m:]])              # [2, 2, 4]

    """

    _passes_required_after: Set[Type[ExportPass]] = {
        ConvertExpandCopyToRepeatPass,
        ConvertSqueezesToViewPass,
        ReplaceScalarWithTensorByProfilePass,
    }

    _TARGET_OPS = {
        exir_ops.edge.aten.index.Tensor,
    }

    @staticmethod
    def _shape_to_stride(
        values_shape: Sequence[int],
    ) -> list[int]:
        strides: list[int] = []
        stride = 1
        for d in reversed(values_shape):
            strides.insert(0, stride)
            stride = stride * d
        return strides

    @staticmethod
    def _validate_tensor_indices(indices):
        assert (
            isinstance(indices, (list, tuple)) and len(indices) > 0
        ), f"index.Tensor expects non-empty indices list/tuple, got {type(indices)}."

        for i, idx in enumerate(indices):
            assert (
                idx is not None
            ), f"index.Tensor: None indices are not supported at the moment (indices[{i}] is None)."
            assert (
                idx.data.dtype == torch.int32
            ), "index.Tensor requires index dtype must be int32"

    def _compute_index_tensor_params(self, x, m, index_shapes):
        """Compute shape/stride-derived parameters needed to lower
        edge.aten.index.Tensor.

        Derives the broadcasted index shape and the scale factors used to flatten and
        acculumulate multi-dimensional indices into a single gather index, following
        the S/W/K/C notation described in the class docstring.

        Args:
          x: Values tensor being indexed.
          m: Number of tensor indices (i.e., len(indices)).
          index_shapes: Shapes corresponding to each tensor index.

        Returns:
          (x_data, S, W, K, C, trailing, lin_scales), where:
            - x_data is `x.data` (FakeTensor)
            - trailing is `x.shape[m:]` as a list of ints
            - lin_scales are per-dimension scale factors for linearization

        """

        x_data = x.data  # FakeTensor
        x_shape = tuple(x_data.shape)
        x_rank = len(x_shape)

        assert x_rank >= 1, f"index.Tensor expects x rank>=1, got {x_shape}."
        assert (
            m <= x_rank
        ), f"index.Tensor has too many indices ({m}) for x rank {x_rank}."

        # Broadcast shape S for indices, and flattened length W
        S = _broadcast_shape(index_shapes)
        W = math.prod(S) if S else 1

        # Compute gather factors K and C for leading-dims indexing
        leading = list(x_shape[:m])
        trailing = list(x_shape[m:])
        K = math.prod(leading) if leading else 1
        C = math.prod(trailing) if trailing else 1

        # Strides for linearization (contiguous-style)
        strides = self._shape_to_stride(x_shape)

        # Stride/C divisibility is guaranteed for contiguous strides and C=prod(trailing).
        lin_scales: list[int] = []
        for i in range(m):
            stride = strides[i]
            lin_scales.append(stride // C)

        return x_data, S, W, K, C, trailing, lin_scales

    def call_operator(self, op, args, kwargs, meta):
        if op not in self._TARGET_OPS:
            return super().call_operator(op, args, kwargs, meta)

        assert (
            len(args) == 2
        ), f"[{self.__class__.__name__}] Expected 2 args for {op}, got {len(args)}."

        x, indices = args

        self._validate_tensor_indices(indices)
        index_shapes = [idx.data.shape for idx in indices]
        m = len(indices)

        x_data, S, W, K, C, trailing, lin_scales = self._compute_index_tensor_params(
            x, m, index_shapes
        )

        (
            view_op,
            unsqueeze_op,
            expand_op,
            mul_scalar_op,
            add_tensor_op,
            to_copy_op,
            tosa_gather_op,
        ) = get_index_tensor_decomposition(op)

        # ---- optional bool -> int8 ----
        x_for_gather = x
        x_dtype = x_data.dtype
        if x_dtype == torch.bool:
            x_for_gather = super().call_operator(
                to_copy_op,
                (x,),
                {"dtype": torch.int8},
                meta,
                updated=True,
            )

        # ---- x: [1, K, C] ----
        x_1kc = super().call_operator(
            view_op,
            (x_for_gather, [1, K, C]),
            {},
            meta,
            updated=True,
        )

        # Build linear index [1, W] from broadcasted indices
        lin_w = None
        plain_meta = meta_without_qparams(meta)
        for i, idx in enumerate(indices):
            idx_data = idx.data
            idx_shape = tuple(idx_data.shape)

            # Align ranks (prepend 1s) so it can be expanded to broadcast shape
            if len(idx_shape) != len(S):
                idx_aligned_shape = [1] * (len(S) - len(idx_shape)) + list(idx_shape)
                idx_aligned = super().call_operator(
                    view_op,
                    (idx, idx_aligned_shape),
                    {},
                    plain_meta,
                    updated=True,
                )
            else:
                idx_aligned = idx

            # Broadcast: idx_aligned -> [*S]
            idx_broadcast = super().call_operator(
                expand_op,
                (idx_aligned, list(S)),
                {},
                plain_meta,
                updated=True,
            )

            # Flatten: [*S] -> [W]
            idx_flat = super().call_operator(
                view_op,
                (idx_broadcast, [W]),
                {},
                plain_meta,
                updated=True,
            )

            # Scale by stride factor lin_scales[i]: [W]
            idx_scaled = super().call_operator(
                mul_scalar_op,
                (idx_flat, lin_scales[i]),
                {},
                plain_meta,
                updated=True,
            )

            # Accumulate into lin_1w: [1, W]
            if lin_w is None:
                lin_w = idx_scaled
            else:
                lin_w = super().call_operator(
                    add_tensor_op,
                    (lin_w, idx_scaled),
                    {},
                    plain_meta,
                    updated=True,
                )

        if lin_w is None:
            raise RuntimeError(
                f"[{self.__class__.__name__}] internal error: lin_1w not constructed."
            )

        # Make indices shape [1, W] for tosa.GATHER
        lin_1w = super().call_operator(
            unsqueeze_op,
            (lin_w, 0),
            {},
            plain_meta,
            updated=True,
        )

        # ---- backend tosa gather ---
        # tosa.GATHER(x=[1,K,C], indices=[1,W]) -> [1,W,C]
        gathered_1wc = super().call_operator(
            tosa_gather_op,
            (x_1kc, lin_1w),
            {},
            meta,
            updated=True,
        )

        # ---- output: [*S, *trailing] ----
        out_shape = list(S) + list(trailing)
        out = super().call_operator(
            view_op,
            (gathered_1wc, out_shape),
            {},
            meta,
            updated=True,
        )

        # ---- optional int8 -> bool ----
        if x_dtype == torch.bool:
            out = super().call_operator(
                to_copy_op,
                (out,),
                {"dtype": torch.bool},
                meta,
                updated=True,
            )

        return out
