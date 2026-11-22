# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""NVFP4 linear/embedding dispatch captured at eager and torch.export time.

Environment variables, read when the model is traced, fix the policy that
is compiled into the program:

* ``ET_CUDA_NVFP4_ACT``: ``auto`` (default) times W4A16 against native W4A4
  per layer when native FP4 is available, and decode (M statically <= 4, by
  default only M = 1, see ``ET_CUDA_NVFP4_SMALL_M``) W4A16 against the W4A8
  dp4a GEMV (int8 activations per 16 elements) otherwise; ``w4a16`` forces
  BF16/FP16 activations, ``w4a8`` forces the W4A8 GEMV for decode (other M run
  BF16/FP16 activations, as ``w4a16``), and ``w4a4`` forces FP4 activations
  (native FP4 required).
* ``ET_CUDA_NVFP4_PREFILL_OUTPUT``: ``input`` (default) keeps the activation
  dtype; ``fp32`` returns FP32 from prefill linears (M not statically <= 4).
* ``ET_CUDA_NVFP4_PREFILL_IMPL``: for M not statically <= 64, ``auto``
  (default) uses the fused kernels with native FP4 (SM120+) and whole-weight
  dequant + matmul otherwise; ``kernel`` / ``dequant`` force one for any M
  not statically <= 4. ``w4a4`` and the FP32 prefill output policy always take
  the fused kernels. The choice is made per exported program from M's upper
  bound. A dynamic-M method tunes the fused kernels at its example M (the
  largest, in the Muse Glimmer exports), so dequant + matmul is also faster
  for its short prompts.
* ``ET_CUDA_NVFP4_SMALL_M``: ``1`` (default) routes ``auto``/``w4a16`` linears
  whose M is in 2..16 (static, or dynamic with an upper bound <= 16), and
  ``w4a8`` ones with M in 5..16, to the small-M bucket ops, which time a rows
  GEMV, W4A16 and (``auto`` with native FP4) W4A4; with native FP4, ``auto``
  M <= 4 keeps ``nvfp4_auto_decode``, and the FP32 prefill output policy and
  an explicit ``ET_CUDA_NVFP4_PREFILL_IMPL=dequant`` keep M 5..16 off them.
  ``0`` keeps the decode and prefill paths.
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
    select_bucket,
)
from executorch.backends.cuda.triton.kernels.nvfp4_output import (
    NVFP4_DECODE_MAX_ROWS,
    NVFP4_FUSED_PREFILL_MAX_ROWS,
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
_DECODE_MODE_SPLITS = tuple(
    mode * _MODE_STRIDE + split for mode in (0, 1, 2) for split in SPLIT_K_CANDIDATES
)
_ACT_MODES = ("auto", "w4a4", "w4a8", "w4a16")


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
    _DECODE_MODE_SPLITS,
    candidate_prepass=_compile_every_candidate_at_once,
    do_bench=nvfp4_launch_bench,
)
def _decode(x, wq, ws, pts, act_scale, unit_scale=False, *, mode_split):
    # Activation mode (0: W4A16, 1: native W4A4, 2: W4A8 without native FP4)
    # * 64 + split-K; explicit splits bypass nested tuners.
    from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import (
        _launch_w4a4_impl,
        nvfp4_gemv,
    )

    mode, split = divmod(mode_split, _MODE_STRIDE)
    if mode == 1:
        if not _has_native_fp4():
            raise InvalidLaunchParam("NVFP4 W4A4 needs native FP4 (SM120+)")
        return _launch_w4a4_impl(
            x, wq, ws, pts, act_scale, _max_rows(x.shape[0]), split_k=split
        )
    if mode == 2:
        # W4A8 serves NVIDIA GPUs without native FP4 (its asm is NVIDIA PTX).
        if _has_native_fp4() or torch.version.hip is not None:
            raise InvalidLaunchParam(
                "NVFP4 auto decode times W4A8 only on NVIDIA without native FP4"
            )
        from executorch.backends.cuda.triton.kernels.nvfp4_w4a8_gemv import (
            launch_w4a8_gemv,
        )

        return launch_w4a8_gemv(x, wq, ws, pts, _max_rows(x.shape[0]), split_k=split)
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
    """Compare the NVFP4 GEMV with native W4A4, or without native FP4 with the
    W4A8 GEMV, in the AOTI launch context."""
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


def _prefill_impl():
    impl = os.environ.get("ET_CUDA_NVFP4_PREFILL_IMPL", "auto")
    if impl not in ("auto", "kernel", "dequant"):
        raise ValueError("ET_CUDA_NVFP4_PREFILL_IMPL must be auto, kernel or dequant")
    return impl


def _dequant_prefill(flat, mode, output_dtype, impl):
    """Whether an NVFP4 linear takes whole-weight dequant + F.linear.

    auto (default): only without native FP4 (SM120+) and for M not statically
    <= ``NVFP4_FUSED_PREFILL_MAX_ROWS``, where the dense matmul beats in-register
    dequant; smaller M keep the fused kernels, which read the packed weight
    once. The choice is made once per exported program, at M's upper bound.
    """
    rows = flat.shape[0]
    if (
        mode == "w4a4"
        or output_dtype is not None
        or statically_known_true(rows <= NVFP4_DECODE_MAX_ROWS)
    ):
        return False
    if impl == "auto":
        return not _has_native_fp4() and not statically_known_true(
            rows <= NVFP4_FUSED_PREFILL_MAX_ROWS
        )
    return impl == "dequant"


def _dequant_linear(flat, w):
    """Whole-weight dequant + F.linear, chunked along N for huge weights."""
    from executorch.backends.cuda.triton.kernels.nvfp4_dequant import nvfp4_dequant

    def rows(i, j):
        part = _weight_rows(w, slice(i, j))
        dense = nvfp4_dequant(part.qdata, part.scale, part.per_tensor_scale, flat.dtype)
        return F.linear(flat, dense)

    return chunked_dequant_linear(flat, w.shape[0], rows)


@torch.library.custom_op("triton::nvfp4_gated_linear", mutates_args={})
def nvfp4_gated_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Gated projection ``silu(x @ gate.T) * (x @ up.T)`` of a CudaNVFP4Tensor.

    ``weight`` rows are ``[gate; up]`` as produced by
    ``CudaNVFP4Tensor.fuse_along_output([gate, up])`` (per-row global scales
    in the same order), so N must be even. With native FP4 and BF16 (or FP32
    under the FP32 prefill policy) activations, outside ``w4a16``/``w4a8``, it
    runs the fused kernels (in ``auto`` only for M not statically <=
    ``NVFP4_DECODE_MAX_ROWS``, whose decode kernels ``F.linear`` times) unless
    ``F.linear`` would take dequant + matmul; otherwise it is ``F.linear``
    (NVFP4 dispatch) followed by SiLU-gating.

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
    if mode not in _ACT_MODES:
        raise ValueError(f"ET_CUDA_NVFP4_ACT must be one of {_ACT_MODES}")
    impl = _prefill_impl()
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    output_dtype = nvfp4_prefill_output_dtype(flat)
    # FP32 activations take the fused kernels only under the FP32 prefill
    # policy at prefill size, like _linear.
    if (
        mode in ("w4a16", "w4a8")
        or (
            mode != "w4a4"
            and statically_known_true(flat.shape[0] <= NVFP4_DECODE_MAX_ROWS)
        )
        or _dequant_prefill(flat, mode, output_dtype, impl)
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


def _small_m_enabled():
    small_m = os.environ.get("ET_CUDA_NVFP4_SMALL_M", "1")
    if small_m not in ("0", "1"):
        raise ValueError("ET_CUDA_NVFP4_SMALL_M must be 0 or 1")
    return small_m == "1"


def _small_m_op(flat, w, mode, output_dtype, impl, enabled):
    """The small-M bucket op and its arguments for this linear, or None.

    ``auto``/``w4a16`` take the buckets for M in 2..16, and ``w4a8`` for M
    above decode size (its W4A8 GEMV serves decode); only ``auto`` may time
    FP4 activations, and only it passes the static activation scale. With
    native FP4, ``auto`` decode-size M keeps ``nvfp4_auto_decode`` (equal or
    faster than the buckets on an RTX 5090 at M = 2..4).
    """
    rows = flat.shape[0]
    small = statically_known_true(rows <= NVFP4_DECODE_MAX_ROWS)
    native_decode = small and mode == "auto" and _has_native_fp4()
    if (
        not enabled
        or native_decode
        or not (mode in ("auto", "w4a16") or (mode == "w4a8" and not small))
        or output_dtype is not None
        or statically_known_true(rows <= 1)
        or _dequant_prefill(flat, mode, output_dtype, impl)
    ):
        return None
    from executorch.backends.cuda.triton.kernels.nvfp4_small_m_gemm import (
        NVFP4_SMALL_M_GEMM,
    )

    fp4 = mode == "auto" and _has_native_fp4()
    op_args = (
        flat,
        w.qdata,
        w.scale,
        w.per_tensor_scale,
        w.act_per_tensor_scale if fp4 else None,
        w.unit_scale,
        mode != "auto",
    )
    bucket = select_bucket(NVFP4_SMALL_M_GEMM, *op_args)
    return None if bucket is None else (NVFP4_SMALL_M_GEMM.op(bucket), op_args)


@CudaNVFP4Tensor.implements([torch.ops.aten.linear.default])
@CudaNVFP4Tensor.implements_torch_function([F.linear])
def _linear(func, types, args, kwargs):  # noqa: C901
    x, w = args[:2]
    bias = args[2] if len(args) > 2 else kwargs.get("bias")
    mode = os.environ.get("ET_CUDA_NVFP4_ACT", "auto")
    if mode not in _ACT_MODES:
        raise ValueError(f"ET_CUDA_NVFP4_ACT must be one of {_ACT_MODES}")
    impl = _prefill_impl()
    use_small_m = _small_m_enabled()
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

        if mode == "w4a4" and not _has_native_fp4():
            raise RuntimeError(
                "Native NVFP4 W4A4 requires an SM120 or newer CUDA target"
            )
        if mode == "w4a8" and torch.version.hip is not None:
            raise RuntimeError("NVFP4 W4A8 (dp4a PTX) requires an NVIDIA GPU")
        small = statically_known_true(flat.shape[0] <= NVFP4_DECODE_MAX_ROWS)
        small_m = _small_m_op(flat, w, mode, output_dtype, impl, use_small_m)
        if mode == "w4a4":
            out = nvfp4_w4a4(
                flat,
                w.qdata,
                w.scale,
                w.per_tensor_scale,
                w.act_per_tensor_scale,
                output_dtype=output_dtype,
            )
        elif small_m is not None:
            op, op_args = small_m
            out = op(*op_args)
        elif small and mode == "auto":
            # Only W4A4 reads the static activation scale.
            out = nvfp4_auto_decode(
                flat,
                w.qdata,
                w.scale,
                w.per_tensor_scale,
                w.act_per_tensor_scale if _has_native_fp4() else None,
                w.unit_scale,
            )
        elif small and mode == "w4a8":
            from executorch.backends.cuda.triton.kernels.nvfp4_w4a8_gemv import (
                nvfp4_w4a8_gemv,
            )

            out = nvfp4_w4a8_gemv(flat, w.qdata, w.scale, w.per_tensor_scale)
        elif small:
            out = nvfp4_gemv(
                flat, w.qdata, w.scale, w.per_tensor_scale, unit_scale=w.unit_scale
            )
        elif _dequant_prefill(flat, mode, output_dtype, impl):
            out = _dequant_linear(flat, w)
        elif mode != "auto" or not _has_native_fp4():
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
