"""Roofline-only actual-row grid for the unchanged native W4A16 producer.

Keep static strides and deterministic reduction. Conditional branch kernels
can be autotuned with the enclosing method's1024-row example even when the
runtime predicate restricts that branch to128/256 rows. Reserve1024 rows for
new bucket workspaces; retain actual-row producer/reducer work. Existing
1..64 operations retain their original launch helper and CUDA graph shapes.
"""
import torch
import triton
from typing import Sequence
from torch.library import wrap_triton
from executorch.backends.cuda.triton.kernels.quantized_gemm_utils import splitk_reduce
from executorch.backends.cuda.triton.kernels.quantized_gemm_family import (
    _QUEUING_ONLY, _parallel_compile_available, _async_compile,
    _queue_untuned_configs, _precompile_untuned_configs, _tuned_kernel,
)


def launch_actual_rows(
    kernel,
    *,
    bucket: int,
    m,
    n: int,
    device: torch.device,
    split_k: int,
    block_m: int,
    inputs: Sequence,
    shape_args: Sequence,
    split_k_on_axis_1: bool = False,
    reduce_block_n: int = 64,
    **constexprs,
) -> torch.Tensor:
    """Keep native workspace and arithmetic; launch only actual M tiles."""
    out = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    if split_k > 1:
        # Inductor may benchmark an unselected cond branch at the enclosing
        # method's row hint. A128-row temporary is unsafe with a1024-row
        # benchmark grid. Keep this pitch constant for AOTI constexpr strides,
        # while actual runtime M continues to determine producer/reducer work.
        partial = torch.empty((split_k, max(1024, bucket), n), dtype=torch.float32, device=device)
        stride_os = partial.stride(0)
    else:
        partial = out
        stride_os = 0

    def grid(meta):
        n_tiles = triton.cdiv(n, meta["BLOCK_N"])
        m_tiles = triton.cdiv(m, meta["BLOCK_M"]) if "BLOCK_M" in meta else 1
        # Retain fixed grid rank for a symbolic M; never branch in Python on
        # whether the runtime row count produces one tile.
        if split_k_on_axis_1:
            return (n_tiles, split_k, m_tiles) if split_k > 1 else (n_tiles, m_tiles)
        return (n_tiles, m_tiles, split_k)

    args = (
        *inputs,
        partial,
        *shape_args,
        stride_os,
        partial.stride(-2),
        partial.stride(-1),
    )
    kwargs = {"SPLIT_K": split_k, **constexprs}
    if _QUEUING_ONLY.get():
        if (
            not _parallel_compile_available()
            or _async_compile.active_mode.get() is None
        ):
            raise RuntimeError("candidate prepass lost its active compile pool")
        _queue_untuned_configs(kernel, args, kwargs)
        # The prepass discards the result, so the output is not written.
        return out
    _precompile_untuned_configs(kernel, args, kwargs)
    wrap_triton(_tuned_kernel(kernel, args, kwargs))[grid](*args, **kwargs)
    if split_k > 1:
        splitk_reduce(partial, out, block_m, reduce_block_n)
    return out


INSTALLED = False

def install():
    global INSTALLED
    if INSTALLED: return
    from executorch.backends.cuda.triton.kernels import int4_quantized_gemm as native
    original = native.launch_split_k_gemm
    def launch(kernel, *, bucket, **kwargs):
        if bucket <= 64:
            return original(kernel, bucket=bucket, **kwargs)
        return launch_actual_rows(kernel, bucket=bucket, **kwargs)
    native.launch_split_k_gemm = launch
    INSTALLED = True
