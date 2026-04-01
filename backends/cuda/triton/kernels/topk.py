# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Triton Top-K Kernel for ExecuTorch CUDA Backend.

Replaces aten.topk with a Triton implementation so the op is compiled
directly into the AOTInductor .so (no C++ fallback shim needed).

Algorithm: iterative argmax/argmin with masking. One program per row,
single thread block loads the entire row (N <= 4096).
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


def _next_power_of_2(x: int) -> int:
    """Return the smallest power of 2 >= x."""
    n = 1
    while n < x:
        n *= 2
    return n


@triton.jit
def _topk_kernel(
    X,
    OUT_V,
    OUT_I,
    stride_xn,
    stride_ovn,
    stride_oin,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    LARGEST: tl.constexpr,
):
    """Single-block topk: one program per row, iterative max/min with masking."""
    pid = tl.program_id(0)
    row_ptr = X + pid * stride_xn
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    if LARGEST:
        FILL: tl.constexpr = float("-inf")
    else:
        FILL: tl.constexpr = float("inf")

    raw_vals = tl.load(row_ptr + offs, mask=mask, other=FILL).to(tl.float32)
    idxs = offs.to(tl.int64)

    # NaN handling: torch.topk treats NaN as larger than all finite values
    # regardless of the `largest` flag. Replace NaN with +inf so NaN sorts
    # to the front for largest=True and to the back for largest=False.
    # Use a consumed mask (not FILL overwrite) to avoid collision between
    # FILL and the NaN replacement value.
    is_nan = raw_vals != raw_vals  # NaN != NaN is true
    vals = tl.where(is_nan, float("inf"), raw_vals)
    consumed = offs >= N  # start with out-of-range positions masked

    out_v_ptr = OUT_V + pid * stride_ovn
    out_i_ptr = OUT_I + pid * stride_oin

    for j in tl.static_range(0, K):
        effective = tl.where(consumed, FILL, vals)

        if LARGEST:
            vsel = tl.max(effective, axis=0)
        else:
            vsel = tl.min(effective, axis=0)

        eq = (effective == vsel) & ~consumed
        # For ties, pick the smallest index: add BLOCK to non-equal positions
        big = tl.where(eq, tl.zeros_like(idxs), tl.zeros_like(idxs) + BLOCK)
        arg = tl.min(idxs + big, axis=0)

        # Restore NaN if the selected element was originally NaN
        was_nan = (
            tl.sum(
                tl.where(idxs == arg, is_nan.to(tl.float32), tl.zeros_like(vals)),
                axis=0,
            )
            > 0
        )
        out_val = tl.where(was_nan, float("nan"), vsel)

        tl.store(out_v_ptr + j, out_val)
        tl.store(out_i_ptr + j, arg)

        consumed = consumed | (idxs == arg)


@triton_op("triton::topk", mutates_args={})
def topk(
    self: torch.Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton top-k implementation.

    Supports arbitrary dim by transposing so the target dimension is last,
    running the kernel, then transposing back.

    Args:
        self: Input tensor on CUDA.
        k: Number of top elements.
        dim: Dimension to operate on (default -1).
        largest: If True return largest, else smallest.
        sorted: If True return in sorted order (inherent in iterative algo).

    Returns:
        (values, indices) tensors with the topk dimension replaced by k.
    """
    # Normalize dim
    ndim = self.dim()
    if dim < 0:
        dim = dim + ndim

    N = self.shape[dim]
    if k < 0:
        raise ValueError(f"k ({k}) must be non-negative")
    if k > N:
        raise ValueError(f"k ({k}) is too big for dimension {dim} of size {N}")
    if N > 4096:
        raise ValueError(
            f"N={N} exceeds max supported size 4096. "
            "The kernel loads entire rows into one thread block."
        )

    # Handle empty dimension
    if N == 0:
        out_shape = list(self.shape)
        out_shape[dim] = 0
        values = self.new_empty(out_shape)
        indices = self.new_empty(out_shape, dtype=torch.int64)
        return values, indices

    # Move target dim to last position for contiguous row access
    if dim != ndim - 1:
        self = self.transpose(dim, ndim - 1).contiguous()
    elif not self.is_contiguous():
        self = self.contiguous()

    # Flatten all batch dims into one
    orig_shape = self.shape
    N = orig_shape[-1]  # row length
    num_rows = self.numel() // N
    x_flat = self.reshape(num_rows, N)

    # Allocate outputs
    values = torch.empty(num_rows, k, dtype=self.dtype, device=self.device)
    indices = torch.empty(num_rows, k, dtype=torch.int64, device=self.device)

    if k == 0 or num_rows == 0:
        # Reshape and transpose back
        out_shape = list(orig_shape)
        out_shape[-1] = k
        values = values.reshape(out_shape)
        indices = indices.reshape(out_shape)
        if dim != ndim - 1:
            values = values.transpose(dim, ndim - 1).contiguous()
            indices = indices.transpose(dim, ndim - 1).contiguous()
        return values, indices

    BLOCK = _next_power_of_2(N)

    grid = (num_rows,)
    wrap_triton(_topk_kernel)[grid](
        x_flat,
        values,
        indices,
        x_flat.stride(0),
        values.stride(0),
        indices.stride(0),
        N=N,
        K=k,
        BLOCK=BLOCK,
        LARGEST=largest,
    )

    # Reshape back to original batch shape with k replacing dim size
    out_shape = list(orig_shape)
    out_shape[-1] = k
    values = values.reshape(out_shape)
    indices = indices.reshape(out_shape)

    # Transpose back if we moved dim
    if dim != ndim - 1:
        values = values.transpose(dim, ndim - 1).contiguous()
        indices = indices.transpose(dim, ndim - 1).contiguous()

    return values, indices


@topk.register_fake
def _topk_abstract(
    self: torch.Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Abstract/fake implementation for torch.export."""
    ndim = self.dim()
    if dim < 0:
        dim = dim + ndim
    out_shape = list(self.shape)
    out_shape[dim] = k
    values = self.new_empty(out_shape)
    indices = self.new_empty(out_shape, dtype=torch.int64)
    return values, indices
