# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""NVFP4 linear/embedding dispatch captured at eager and torch.export time.

Environment variables, read when the model is traced, fix the policy that
is compiled into the program:

* ``ET_CUDA_NVFP4_ACT``: ``auto`` (default) times W4A16 against native W4A4
  per layer when native FP4 is available, ``w4a16`` forces weight-only, and
  ``w4a4`` forces FP4 activations (native FP4 required).
* ``ET_CUDA_NVFP4_PREFILL_OUTPUT``: ``input`` (default) keeps the activation
  dtype; ``fp32`` returns FP32 from prefill linears (M not statically <= 4).
* ``ET_CUDA_NVFP4_SOFT_W4A8``: ``1`` (default) lets ``auto`` decode without
  native FP4 time the W4A8 dp4a GEMV against the W4A16 GEMV; ``0`` keeps W4A16.
* ``ET_CUDA_NVFP4_PREFILL_IMPL``: ``auto`` (default) uses the fused kernels
  with native FP4 and dequant + matmul otherwise; ``kernel`` / ``dequant``
  force one.
"""

import os

import torch
import torch.nn.functional as F

from executorch.backends.cuda.autotune.launch_params import (
    autotune_launch_param,
    InvalidLaunchParam,
)
from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor
from executorch.backends.cuda.quantize_op_dispatch._gemm_family_dispatch import (
    can_launch_triton,
    chunked_dequant_linear,
)
from executorch.backends.cuda.triton.kernels.nvfp4_output import (
    NVFP4_DECODE_MAX_ROWS,
    nvfp4_prefill_output_dtype,
)
from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import (
    _check_operands,
    _check_static_scale,
    _has_native_fp4_conversion as _has_native_fp4,
    _max_rows,
    SPLIT_K_CANDIDATES,
)
from executorch.backends.cuda.triton.kernels.nvfp4_tuning import nvfp4_launch_bench
from executorch.backends.cuda.triton.kernels.quantized_gemm_family import (
    _compile_every_candidate_at_once,
)
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.library import triton_op

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
def _prefill(x, wq, ws, pts, act_scale, max_rows, *, mode_split, output_dtype=None):
    # max_rows is the traced M's bound: timing (at hinted sizes) rejects the
    # same splits as the final traced call.
    from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import (
        _check_split,
        _launch_w4a4_impl,
        nvfp4_w4a16,
    )

    fp4, split = divmod(mode_split, _MODE_STRIDE)
    _check_split(x.shape[-1], split, m=max_rows, n=wq.shape[0])
    if fp4:
        if not _has_native_fp4():
            raise InvalidLaunchParam("NVFP4 W4A4 needs native FP4 (SM120+)")
        # Unchecked launch: the op validated its operands, and timing runs
        # this on random stand-ins.
        return _launch_w4a4_impl(
            x,
            wq,
            ws,
            pts,
            act_scale,
            max_rows,
            split_k=split,
            output_dtype=output_dtype,
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
        _launch_w4a4_impl,
        nvfp4_gemv,
    )

    fp4, split = divmod(mode_split, _MODE_STRIDE)
    if fp4:
        if not _has_native_fp4():
            raise InvalidLaunchParam("NVFP4 W4A4 needs native FP4 (SM120+)")
        return _launch_w4a4_impl(
            x, wq, ws, pts, act_scale, _max_rows(x.shape[0]), split_k=split
        )
    return nvfp4_gemv(x, wq, ws, pts, split_k=split, unit_scale=unit_scale)


@triton_op("triton::nvfp4_auto_prefill", mutates_args={})
def nvfp4_auto_prefill(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    act_scale: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Keep mode selection inside the op so AOTI can time both precisions."""
    _check_operands(x, qdata, scale, per_tensor_scale)
    _check_static_scale(act_scale, x)
    kwargs = {} if output_dtype is None else {"output_dtype": output_dtype}
    return _prefill(
        x,
        qdata,
        scale,
        per_tensor_scale,
        act_scale,
        _max_rows(x.shape[0]),
        **kwargs,
    )


@triton_op("triton::nvfp4_auto_decode", mutates_args={})
def nvfp4_auto_decode(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    act_scale: torch.Tensor | None = None,
    unit_scale: bool = False,
) -> torch.Tensor:
    """Compare NVFP4 GEMV and native FP4 in the AOTI launch context."""
    _check_operands(x, qdata, scale, per_tensor_scale)
    _check_static_scale(act_scale, x)
    return _decode(x, qdata, scale, per_tensor_scale, act_scale, unit_scale)


@autotune_launch_param(
    "mode_split",
    _MODE_SPLITS,
    candidate_prepass=_compile_every_candidate_at_once,
    do_bench=nvfp4_launch_bench,
)
def _prefill_gate_up(
    x, wq, ws, pts, act_scale, max_rows, *, mode_split, output_dtype=None
):
    from executorch.backends.cuda.triton.kernels.nvfp4_gated_gemm import (
        _launch_w4a4_gate_up_impl,
        nvfp4_gate_up_epilogue,
    )
    from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import (
        _check_split,
        nvfp4_w4a16,
    )

    fp4, split = divmod(mode_split, _MODE_STRIDE)
    _check_split(x.shape[-1], split, m=max_rows, n=wq.shape[0])
    if fp4:
        if not _has_native_fp4():
            raise InvalidLaunchParam("NVFP4 W4A4 needs native FP4 (SM120+)")
        return _launch_w4a4_gate_up_impl(
            x,
            wq,
            ws,
            pts,
            act_scale,
            max_rows,
            split_k=split,
            output_dtype=output_dtype,
        )
    return nvfp4_gate_up_epilogue(
        nvfp4_w4a16(x, wq, ws, pts, split_k=split, output_dtype=output_dtype)
    )


@triton_op("triton::nvfp4_auto_gate_up_prefill", mutates_args={})
def nvfp4_auto_gate_up_prefill(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    act_scale: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    _check_operands(x, qdata, scale, per_tensor_scale)
    _check_static_scale(act_scale, x)
    kwargs = {} if output_dtype is None else {"output_dtype": output_dtype}
    return _prefill_gate_up(
        x,
        qdata,
        scale,
        per_tensor_scale,
        act_scale,
        _max_rows(x.shape[0]),
        **kwargs,
    )


def _nvfp4_prefill_impl():
    """Prefill (non-small M) NVFP4 linear implementation.

    auto (default): fused W4A4/W4A16 kernels where FP4 tensor cores exist
    (SM120+); elsewhere whole-weight dequant + F.linear, the integer GGUF
    formats' M > 64 path, which beats in-register dequant there.
    """
    impl = os.environ.get("ET_CUDA_NVFP4_PREFILL_IMPL", "auto")
    if impl not in ("auto", "kernel", "dequant"):
        raise ValueError("ET_CUDA_NVFP4_PREFILL_IMPL must be auto, kernel or dequant")
    if impl == "auto":
        return "kernel" if _has_native_fp4() else "dequant"
    return impl


def _dequant_linear(flat, w):
    """Whole-weight dequant + F.linear, chunked along N for huge weights."""
    from executorch.backends.cuda.triton.kernels.nvfp4_dequant import nvfp4_dequant

    pts = w.per_tensor_scale
    row_pts = pts.numel() != 1

    def rows(i, j):
        dense = nvfp4_dequant(
            w.qdata[i:j], w.scale[i:j], pts[i:j] if row_pts else pts, flat.dtype
        )
        return F.linear(flat, dense)

    return chunked_dequant_linear(flat, w.shape[0], rows)


@torch.library.custom_op("triton::nvfp4_gated_linear", mutates_args={})
def nvfp4_gated_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Gated projection ``silu(x @ gate.T) * (x @ up.T)`` of a CudaNVFP4Tensor.

    ``weight`` rows are ``[gate; up]`` as produced by
    ``CudaNVFP4Tensor.fuse_along_output([gate, up])`` (per-row global scales
    in the same order), so N must be even. With native FP4 and BF16 (or FP32
    under the FP32 prefill policy) activations, outside ``w4a16`` mode, it runs
    the fused prefill kernels; otherwise it is ``F.linear`` (NVFP4 dispatch,
    including decode kernels for small M) followed by SiLU-gating.

    The weight stays an op argument until subclass decomposition: accessing a
    parametrized NVFP4 weight's fields in a model forward makes Dynamo capture
    anonymous constants, which cannot share GPU storage with the decode
    method's named parameters.
    """
    raise TypeError("NVFP4 gated linear requires a CudaNVFP4Tensor weight")


@nvfp4_gated_linear.register_fake
def _fake_gated_linear(x, weight):
    raise TypeError("NVFP4 gated linear requires a CudaNVFP4Tensor weight")


@CudaNVFP4Tensor.implements([torch.ops.triton.nvfp4_gated_linear.default])
def _gated_linear(func, types, args, kwargs):
    x, weight = args
    if weight.shape[0] % 2 or x.shape[-1] != weight.shape[1]:
        raise ValueError("NVFP4 gated linear requires even N and matching K")
    mode = os.environ.get("ET_CUDA_NVFP4_ACT", "auto")
    if mode not in ("auto", "w4a4", "w4a16"):
        raise ValueError("ET_CUDA_NVFP4_ACT must be auto, w4a4, or w4a16")
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    output_dtype = nvfp4_prefill_output_dtype(flat)
    # FP32 activations take the fused kernels only under the FP32 prefill
    # policy at prefill size, like _linear.
    if (
        mode == "w4a16"
        or _nvfp4_prefill_impl() == "dequant"
        or (
            x.dtype != torch.bfloat16
            and not (x.dtype == torch.float32 and output_dtype == torch.float32)
        )
        or not can_launch_triton(flat)
        or not _has_native_fp4()
    ):
        projected = F.linear(x, weight)
        half = projected.shape[-1] // 2
        return F.silu(projected[..., :half]) * projected[..., half:]
    from executorch.backends.cuda.triton.kernels.nvfp4_gated_gemm import (
        nvfp4_w4a4_gate_up,
    )

    operation = nvfp4_w4a4_gate_up if mode == "w4a4" else nvfp4_auto_gate_up_prefill
    out = operation(
        flat,
        weight.qdata,
        weight.scale,
        weight.per_tensor_scale,
        weight.act_per_tensor_scale,
        output_dtype=output_dtype,
    )
    return out.reshape(*x.shape[:-1], weight.shape[0] // 2)


def _weight_rows(w, rows):
    """Rows ``rows`` (a slice or index tensor) of ``w`` as a CudaNVFP4Tensor."""
    pts = w.per_tensor_scale
    return CudaNVFP4Tensor(
        w.qdata[rows],
        w.scale[rows],
        pts if pts.numel() == 1 else pts[rows],
        orig_dtype=w.orig_dtype,
    )


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
        dtype = output_dtype or x.dtype
        flat = flat.to(dtype)
        out = chunked_dequant_linear(
            flat,
            w.shape[0],
            lambda i, j: F.linear(flat, _weight_rows(w, slice(i, j)).dequantize(dtype)),
        )
    else:
        from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import (
            nvfp4_gemv,
            nvfp4_w4a16,
            nvfp4_w4a4,
        )

        requested_mode = mode
        if not _has_native_fp4():
            if mode == "w4a4":
                raise RuntimeError(
                    "Native NVFP4 W4A4 requires an SM120 or newer CUDA target"
                )
            mode = "w4a16"
        small = statically_known_true(flat.shape[0] <= NVFP4_DECODE_MAX_ROWS)
        if mode == "w4a4":
            out = nvfp4_w4a4(
                flat,
                w.qdata,
                w.scale,
                w.per_tensor_scale,
                w.act_per_tensor_scale,
                output_dtype=output_dtype,
            )
        elif small and mode == "auto":
            out = nvfp4_auto_decode(
                flat,
                w.qdata,
                w.scale,
                w.per_tensor_scale,
                w.act_per_tensor_scale,
                w.unit_scale,
            )
        elif (
            small
            and requested_mode == "auto"
            and not _has_native_fp4()
            and os.environ.get("ET_CUDA_NVFP4_SOFT_W4A8", "1") == "1"
        ):
            from executorch.backends.cuda.triton.kernels.nvfp4_w4a8_gemv import (
                nvfp4_auto_decode_soft,
            )

            out = nvfp4_auto_decode_soft(
                flat, w.qdata, w.scale, w.per_tensor_scale, w.unit_scale
            )
        elif small:
            out = nvfp4_gemv(
                flat, w.qdata, w.scale, w.per_tensor_scale, unit_scale=w.unit_scale
            )
        elif (
            _nvfp4_prefill_impl() == "dequant"
            and output_dtype is None
            and x.dtype in (torch.bfloat16, torch.float16)
        ):
            out = _dequant_linear(flat, w)
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
                w.act_per_tensor_scale,
                output_dtype=output_dtype,
            )
    out = out.reshape(*x.shape[:-1], w.shape[0])
    return out if bias is None else out + bias


@CudaNVFP4Tensor.implements([torch.ops.aten.embedding.default])
@CudaNVFP4Tensor.implements_torch_function([F.embedding])
def _embedding(func, types, args, kwargs):
    if func is F.embedding:
        indices, w = args[:2]
        # F.embedding passes everything after weight by keyword.
        if kwargs.get("max_norm") is not None:
            raise ValueError("NVFP4 embedding does not support in-place max_norm")
    else:
        w, indices = args[:2]
    if not can_launch_triton(indices):
        flat = indices.reshape(-1)
        if flat.numel() and bool(((flat < 0) | (flat >= w.shape[0])).any()):
            raise IndexError(
                f"NVFP4 embedding index out of range for {w.shape[0]} rows"
            )
        return _weight_rows(w, flat).dequantize().reshape(*indices.shape, w.shape[1])
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
