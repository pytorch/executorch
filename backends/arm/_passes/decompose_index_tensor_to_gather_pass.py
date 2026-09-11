# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from itertools import zip_longest
from typing import Sequence, Set, Type

import torch

from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.backends.arm._passes.arm_pass_utils import (
    get_param_tensor,
    is_param_node,
    meta_without_qparams,
)
from executorch.backends.arm._passes.convert_expand_copy_to_repeat import (
    ConvertExpandCopyToRepeatPass,
)
from executorch.backends.arm._passes.convert_squeezes_to_view import (
    ConvertSqueezesToViewPass,
)
from executorch.backends.arm._passes.replace_scalar_with_tensor_pass import (
    ReplaceScalarWithTensorByProfilePass,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


def get_index_tensor_decomposition(op):
    """Return operators used to lower index.Tensor through TOSA gather.

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
    """Compute the broadcasted shape using PyTorch/NumPy semantics.

    Requirements:
      - static shape only
      - shapes are right-aligned; lower-rank shapes are implicitly
        front-padded with 1s
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


class DecomposeIndexTensorToGatherPass(ArmOpTargetedPass):
    """Decompose edge.aten.index.Tensor into a TOSA gather and arithmetic.

    Supported subset:
      y = x.index([None, ..., None, i0, i1, ..., i{m-1}])

    where each ik is a Tensor index, m is the number of index tensors, and the
    optional leading None entries preserve dimensions before the indexed block.

    Constraints:
      - `indices` contains an optional leading run of None entries followed by
        only Tensor indices
      - Each index tensor dtype is int32
      - Index tensor shapes are broadcastable to a common shape `S` (per
        index.Tensor semantics)
      - The `m` tensor indices select one contiguous block of dimensions after
        the leading preserved dimensions.
      - Static shapes are required
      - If `x` has more than 2^31 elements, the computed linear index may
        overflow int32.

    Lowering strategy (single gather)
    ---------------------------------
    Let:
      - `p` be the number of leading None entries
      - `S` be the broadcasted index shape
      - `W = prod(S)` (number of indexed positions)
      - `P = prod(x.shape[:p])` (flattened size of the leading preserved
        dimensions)
      - `K = prod(x.shape[p:p+m])` (flattened size of the indexed block)
      - `C = prod(x.shape[p+m:])` (flattened size of the trailing slice per
        index)
      - `leading = x.shape[:p]` and `trailing = x.shape[p+m:]`

    Steps:
    1) Compute parameters needed to lower index.Tensor
         - `S`, `W`, `P`, `K`, `C`, `leading`, `trailing`
         - `lin_scales` as the contiguous strides of the indexed block
           `x.shape[p:p+m]`.
    2) Reshape x to `[P, K, C]` (`x_pkc`).
    3) Build linear indices by scaling and accumulating the flattened index
       tensors element-wise:
         For each tensor index, broadcast it to `S`, then flatten it:

           idx_broadcast[i] = broadcast_to(indices[p + i], S)
           idx_flat[i] = reshape(idx_broadcast[i], [W])

         For each j in [0, W):

           lin_w[j] =
               sum_{i=0..m-1} idx_flat[i][j] * lin_scales[i]

         Equivalently, in tensor notation:

           lin_w =
               sum_{i=0..m-1} idx_flat[i] * lin_scales[i]  # shape [W]

         Then:

           lin_1w = unsqueeze(lin_w, 0)                    # [1, W]
           lin_pw = lin_1w
           if P > 1:
               lin_pw = expand(lin_1w, [P, W])             # [P, W]
    4) Single gather:
         `tosa.GATHER(x=x_pkc, indices=lin_pw) -> [P,W,C]`
    5) Reshape result to `[*leading, *S, *trailing]`.

    Example:
    Consider:
        x.shape = [2, 3, 4, 5]
        indices = [None, i0, i1]   # p = 1, m = 2
        i0.shape = [2, 1]
        i1.shape = [1, 2]

    This corresponds to ``x[:, i0, i1, :]``: the first dimension is
    preserved, the next two dimensions are indexed, and the last dimension is
    trailing.

    1) The index shapes broadcast to:
        S := [2, 2]
        W := prod(S) = 4

    We preserve p=1 leading dimension and index the next m=2 dimensions:
        leading := x.shape[:p] = [2]
        trailing := x.shape[p+m:] = [5]
        P := prod(leading) = 2
        K := prod(x.shape[p:p+m]) = 3 * 4 = 12
        C := prod(trailing) = 5

    The indexed block has shape [3, 4], so its contiguous strides are:
        lin_scales := [4, 1]

    2) Values are reshaped to:
        x_pkc = view(x, [P, K, C]) = [2, 12, 5]

    3) After broadcasting and flattening the indices to length W:
        i0_broadcast, i1_broadcast have shape S=[2,2]
        i0_flat, i1_flat have shape [W]=[4]

    Linear indices are computed element-wise as:
        for each j in [0, W):

            lin_w[j] = 4 * i0_flat[j] + i1_flat[j]

        hence:

            lin_w = 4 * i0_flat + i1_flat  # shape [W]

        lin_w is then unsqueezed to [1, W] and expanded so that
        lin_pw.shape = [P, W] = [2, 4].

    4) Single Gather:
        out_pwc = tosa.GATHER(values=x_pkc, indices=lin_pw)  # [2, 4, 5]

    5) Reshape result:
        out = view(out_pwc, [*leading, *S, *trailing])       # [2, 2, 2, 5]

    """

    _passes_required_after: Set[Type[ExportPass]] = {
        ConvertExpandCopyToRepeatPass,
        ConvertSqueezesToViewPass,
        ReplaceScalarWithTensorByProfilePass,
    }

    target_ops = {
        exir_ops.edge.aten.index.Tensor,
    }

    def __init__(
        self, exported_program: ExportedProgram | None = None, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.exported_program = exported_program

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
    def _validate_and_split_indices(indices):
        assert (
            isinstance(indices, (list, tuple)) and len(indices) > 0
        ), f"index.Tensor expects non-empty indices list/tuple, got {type(indices)}."

        leading_rank = 0
        while leading_rank < len(indices) and indices[leading_rank] is None:
            leading_rank += 1

        tensor_indices = indices[leading_rank:]
        assert tensor_indices, "index.Tensor expects at least one tensor index."
        for i, idx in enumerate(tensor_indices, start=leading_rank):
            assert idx is not None, (
                "index.Tensor supports None entries only before all tensor indices "
                f"(indices[{i}] is None)."
            )
            assert (
                idx.data.dtype == torch.int32
            ), "index.Tensor requires index dtype must be int32"

        return leading_rank, tensor_indices

    def _compute_index_tensor_params(self, x, leading_rank, m, index_shapes):
        """Compute parameters needed to lower edge.aten.index.Tensor.

        Derives the broadcasted index shape and the scale factors used to
        flatten and accumulate multi-dimensional indices into a single gather
        index, following the S/W/P/K/C notation described in the class
        docstring.

        Args:
            x (ProxyValue): Values tensor being indexed.
            leading_rank (int): Number of leading dimensions preserved by None
                entries.
            m (int): Number of tensor indices.
            index_shapes (Sequence[Sequence[int]]): Shapes corresponding to
                each tensor index.

        Returns:
            tuple: `(x_data, S, W, P, K, C, leading, trailing, lin_scales)`,
                where `x_data` is `x.data`, `leading` and `trailing` contain
                the preserved dimensions, and `lin_scales` contains the
                indexed-block strides used for linearization.

        """
        x_data = x.data  # FakeTensor
        x_shape = tuple(x_data.shape)
        x_rank = len(x_shape)

        assert x_rank >= 1, f"index.Tensor expects x rank>=1, got {x_shape}."
        assert leading_rank + m <= x_rank, (
            "index.Tensor has more preserved and indexed dimensions "
            f"({leading_rank + m}) than the input rank ({x_rank})."
        )

        # Broadcast shape S for indices, and flattened length W
        S = _broadcast_shape(index_shapes)
        W = math.prod(S) if S else 1

        # Compute gather factors for the preserved, indexed, and trailing blocks.
        leading = list(x_shape[:leading_rank])
        indexed = list(x_shape[leading_rank : leading_rank + m])
        trailing = list(x_shape[leading_rank + m :])
        P = math.prod(leading) if leading else 1
        K = math.prod(indexed) if indexed else 1
        C = math.prod(trailing) if trailing else 1

        lin_scales = self._shape_to_stride(indexed)

        return x_data, S, W, P, K, C, leading, trailing, lin_scales

    def _decompose_constant_index(self, x, indices, meta):
        tensor_indices = [
            (dim, index) for dim, index in enumerate(indices) if index is not None
        ]
        if (
            self.exported_program is None
            or len(tensor_indices) != 1
            or any(not isinstance(size, int) for size in x.data.shape)
        ):
            return None

        indexed_dim, index_tensor = tensor_indices[0]
        if not is_param_node(self.exported_program, index_tensor.node):
            return None

        constant_index = get_param_tensor(self.exported_program, index_tensor.node)
        if (
            constant_index is None
            or constant_index.dim() != 1
            or constant_index.numel() == 0
        ):
            return None

        indexed_dim_size = x.data.shape[indexed_dim]
        index_values = []
        for value in constant_index.tolist():
            normalized_value = value if value >= 0 else value + indexed_dim_size
            if normalized_value < 0 or normalized_value >= indexed_dim_size:
                raise IndexError(
                    f"index {value} is out of bounds for dimension {indexed_dim} "
                    f"with size {indexed_dim_size}"
                )
            index_values.append(normalized_value)

        if index_values == list(
            range(index_values[0], index_values[0] + len(index_values))
        ):
            return super().call_operator(
                exir_ops.edge.aten.slice_copy.Tensor,
                (x, indexed_dim, index_values[0], index_values[-1] + 1),
                {},
                meta,
                updated=True,
            )

        slices = []
        for index_value in index_values:
            slices.append(
                super().call_operator(
                    exir_ops.edge.aten.slice_copy.Tensor,
                    (x, indexed_dim, index_value, index_value + 1),
                    {},
                    meta,
                    updated=True,
                )
            )
        return super().call_operator(
            exir_ops.edge.aten.cat.default,
            (slices, indexed_dim),
            {},
            meta,
            updated=True,
        )

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.target_ops:
            return super().call_operator(op, args, kwargs, meta)

        assert (
            len(args) == 2
        ), f"[{self.__class__.__name__}] Expected 2 args for {op}, got {len(args)}."

        x, indices = args

        tensor_indices = [index for index in indices if index is not None]
        if len(tensor_indices) == 1 and tensor_indices[0].data.dtype in (
            torch.bool,
            torch.uint8,
        ):
            return super().call_operator(op, args, kwargs, meta)

        constant_result = self._decompose_constant_index(x, indices, meta)
        if constant_result is not None:
            return constant_result

        leading_rank, indices = self._validate_and_split_indices(indices)
        index_shapes = [idx.data.shape for idx in indices]
        m = len(indices)

        (
            x_data,
            S,
            W,
            P,
            K,
            C,
            leading,
            trailing,
            lin_scales,
        ) = self._compute_index_tensor_params(x, leading_rank, m, index_shapes)

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

        # ---- x: [P, K, C] ----
        x_pkc = super().call_operator(
            view_op,
            (x_for_gather, [P, K, C]),
            {},
            meta,
            updated=True,
        )

        # Build linear index [W] from broadcasted indices
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

            # Accumulate into lin_w: [W]
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
                f"[{self.__class__.__name__}] internal error: lin_w not constructed."
            )

        # Make indices shape [P, W] for tosa.GATHER.
        lin_1w = super().call_operator(
            unsqueeze_op,
            (lin_w, 0),
            {},
            plain_meta,
            updated=True,
        )
        lin_pw = lin_1w
        if P > 1:
            lin_pw = super().call_operator(
                expand_op,
                (lin_1w, [P, W]),
                {},
                plain_meta,
                updated=True,
            )

        # ---- backend tosa gather ---
        # tosa.GATHER(x=[P,K,C], indices=[P,W]) -> [P,W,C]
        gathered_pwc = super().call_operator(
            tosa_gather_op,
            (x_pkc, lin_pw),
            {},
            meta,
            updated=True,
        )

        # ---- output: [*leading, *S, *trailing] ----
        out_shape = list(leading) + list(S) + list(trailing)
        out = super().call_operator(
            view_op,
            (gathered_pwc, out_shape),
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
