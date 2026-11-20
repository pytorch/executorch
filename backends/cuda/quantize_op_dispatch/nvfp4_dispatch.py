# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""NVFP4 linear/embedding dispatch captured at eager and torch.export time."""

import os

import torch
import torch.nn.functional as F

from executorch.backends.cuda.autotune.launch_params import autotune_launch_param
from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor
from executorch.backends.cuda.quantize_op_dispatch._gemm_family_dispatch import (
    can_launch_triton,
)
from executorch.backends.cuda.triton.kernels.nvfp4_output import (
    nvfp4_prefill_fp32_enabled,
    nvfp4_prefill_output_dtype,
)
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.library import triton_op


@torch._dynamo.assume_constant_result
def _has_native_fp4():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 12


from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import (
    SPLIT_K_CANDIDATES,
)
from executorch.backends.cuda.triton.kernels.nvfp4_tuning import nvfp4_launch_bench
from executorch.backends.cuda.triton.kernels.quantized_gemm_family import (
    _compile_every_candidate_at_once,
)

_MODE_STRIDE = 64
_MODE_SPLITS = tuple(
    mode * _MODE_STRIDE + split for mode in (0, 1) for split in SPLIT_K_CANDIDATES
)


@autotune_launch_param(
    "mode_split",
    _MODE_SPLITS,
    candidate_prepass=_compile_every_candidate_at_once,
    do_bench=nvfp4_launch_bench,
)
def _prefill(x, wq, ws, pts, act_scale, *, mode_split, output_dtype=None):
    from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import (
        _check_split,
        nvfp4_w4a16,
        nvfp4_w4a4,
    )

    fp4, split = divmod(mode_split, _MODE_STRIDE)
    _check_split(x.shape[-1], split)
    if fp4:
        return nvfp4_w4a4(
            x, wq, ws, pts, act_scale, split_k=split, output_dtype=output_dtype
        )
    return nvfp4_w4a16(x, wq, ws, pts, split_k=split, output_dtype=output_dtype)


@autotune_launch_param(
    "mode_split",
    _MODE_SPLITS,
    candidate_prepass=_compile_every_candidate_at_once,
    do_bench=nvfp4_launch_bench,
)
def _decode(x, wq, ws, pts, act_scale, unit_scale=False, *, mode_split):
    # Activation mode * 64 + split-K; explicit splits bypass nested tuners.
    from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import (
        _check_split,
        nvfp4_gemv,
        nvfp4_w4a4,
    )

    fp4, split = divmod(mode_split, _MODE_STRIDE)
    if fp4:
        _check_split(x.shape[-1], split)
        return nvfp4_w4a4(x, wq, ws, pts, act_scale, split_k=split)
    return nvfp4_gemv(x, wq, ws, pts, split_k=split, unit_scale=unit_scale)


@triton_op("triton::nvfp4_auto_prefill", mutates_args={})
def nvfp4_auto_prefill(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    act_scale: float = 0.0,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Keep mode selection inside the op so AOTI can time both precisions."""
    kwargs = {} if output_dtype is None else {"output_dtype": output_dtype}
    return _prefill(x, qdata, scale, per_tensor_scale, act_scale, **kwargs)


@triton_op("triton::nvfp4_auto_decode", mutates_args={})
def nvfp4_auto_decode(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    act_scale: float = 0.0,
    unit_scale: bool = False,
) -> torch.Tensor:
    """Compare NVFP4 GEMV and native FP4 in the AOTI launch context."""
    return _decode(x, qdata, scale, per_tensor_scale, act_scale, unit_scale)


@CudaNVFP4Tensor.implements([torch.ops.aten.linear.default])
@CudaNVFP4Tensor.implements_torch_function([F.linear])
def _linear(func, types, args, kwargs):
    x, w = args[:2]
    bias = args[2] if len(args) > 2 else kwargs.get("bias")
    mode = os.environ.get("ET_CUDA_NVFP4_ACT", "auto")
    if mode not in ("auto", "w4a4", "w4a16"):
        raise ValueError("ET_CUDA_NVFP4_ACT must be auto, w4a4, or w4a16")
    if x.shape[-1] != w.shape[-1]:
        raise RuntimeError("NVFP4 linear input and weight K dimensions must match")
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    output_dtype = nvfp4_prefill_output_dtype(flat)
    allowed_dtype = x.dtype in (torch.bfloat16, torch.float16) or (
        x.dtype == torch.float32 and output_dtype == torch.float32
    )
    if not can_launch_triton(flat) or not allowed_dtype:
        out = F.linear(flat, w.dequantize(x.dtype))
    else:
        from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import (
            nvfp4_gemv,
            nvfp4_w4a16,
            nvfp4_w4a4,
        )

        if not _has_native_fp4():
            if mode == "w4a4":
                raise RuntimeError(
                    "Native NVFP4 W4A4 requires an SM120 or newer CUDA target"
                )
            mode = "w4a16"
        small = statically_known_true(flat.shape[0] <= 4)
        if mode == "w4a4":
            out = nvfp4_w4a4(
                flat,
                w.qdata,
                w.scale,
                w.per_tensor_scale,
                w.act_per_tensor_scale or 0.0,
                output_dtype=output_dtype,
            )
        elif small and mode == "auto":
            out = nvfp4_auto_decode(
                flat,
                w.qdata,
                w.scale,
                w.per_tensor_scale,
                w.act_per_tensor_scale or 0.0,
                w.unit_scale,
            )
        elif small:
            out = nvfp4_gemv(
                flat, w.qdata, w.scale, w.per_tensor_scale, unit_scale=w.unit_scale
            )
        elif mode == "w4a16":
            out = nvfp4_w4a16(
                flat, w.qdata, w.scale, w.per_tensor_scale, output_dtype=output_dtype
            )
        else:
            out = nvfp4_auto_prefill(
                flat,
                w.qdata,
                w.scale,
                w.per_tensor_scale,
                w.act_per_tensor_scale or 0.0,
                output_dtype=output_dtype,
            )
    out = out.reshape(*x.shape[:-1], w.shape[0])
    return out if bias is None else out + bias


@CudaNVFP4Tensor.implements([torch.ops.aten.embedding.default])
@CudaNVFP4Tensor.implements_torch_function([F.embedding])
def _embedding(func, types, args, kwargs):
    if func is F.embedding:
        indices, w = args[:2]
        max_norm = args[3] if len(args) > 3 else kwargs.get("max_norm")
        if max_norm is not None:
            raise ValueError("NVFP4 embedding does not support in-place max_norm")
    else:
        w, indices = args[:2]
    if not can_launch_triton(indices):
        return F.embedding(indices, w.dequantize())
    from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import (
        nvfp4_embedding,
    )

    return nvfp4_embedding(
        indices.contiguous(), w.qdata, w.scale, w.per_tensor_scale, w.orig_dtype
    )


@CudaNVFP4Tensor.implements(
    [torch.ops.aten.detach.default, torch.ops.aten.alias.default]
)
def _identity(func, types, args, kwargs):
    return args[0]


@CudaNVFP4Tensor.implements([torch.ops.aten.t.default])
def _transpose(func, types, args, kwargs):
    return args[0].dequantize().t()


@CudaNVFP4Tensor.implements([torch.ops.aten._to_copy.default])
def _copy(func, types, args, kwargs):
    return args[0].to(**kwargs)
