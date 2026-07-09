#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the GGUF Q4_K / Q5_K / Q6_K linear lowering.

A linear whose weight is an ``ExportableGGUFTensor`` (extension/llm/export/gguf)
exports to ``linear(x, torchao::dequantize_gguf(weight, ggml_type, ...), bias)``.
The MLX ``GGUF_QUANTIZED_LINEAR`` pattern (custom_kernel_ops/gguf/patterns.py)
matches that subgraph and lowers it to fused Metal kernels (mat-vec for decode,
mat-mat for prefill). These tests compare the fused kernels against the eager
reference (``gguf``-package dequant + ``F.linear``) on the same packed weight,
so quantization quality is irrelevant -- only kernel-vs-reference numerics are
checked.

``GGUFLinearDynamicTest`` exports once with a symbolic seqlen and runs the same
.pte with M=1 and M>1 to exercise both branches of the runtime ``IfNode``
(decode mat-vec vs prefill mat-mat).

Usage::

    python -m executorch.backends.mlx.custom_kernel_ops.gguf.test.test_linear run
    python -m executorch.backends.mlx.custom_kernel_ops.gguf.test.test_linear run -v
    python -m executorch.backends.mlx.custom_kernel_ops.gguf.test.test_linear list
"""

import os
from contextlib import contextmanager
from typing import List, Tuple

# Importing the patterns module registers GGUF_QUANTIZED_LINEAR / _EMBEDDING.
import executorch.backends.mlx.custom_kernel_ops.gguf.patterns  # noqa: F401
import torch
import torch.nn as nn
from executorch.backends.mlx.custom_kernel_ops.gguf.q5k import Q5K_BLOCK_BYTES
from executorch.backends.mlx.custom_kernel_ops.gguf.q6k import Q6K_BLOCK_BYTES, QK_K
from executorch.backends.mlx.test.test_utils import OpTestCase
from executorch.extension.llm.export.gguf import ExportableGGUFTensor


# ---------------------------------------------------------------------------
# GGUF Q6_K / Q4_K test fixtures.
#
# The Python ``gguf`` package can dequantize Q6_K but does NOT implement Q6_K
# quantization, so we build the packed weight here. Quantization quality is
# irrelevant: the tests only compare the kernel against the eager reference on
# the *same* bytes, so we just emit valid random blocks (random ql/qh/scales
# plus a small finite fp16 ``d`` -- the one field that must be finite).
# ---------------------------------------------------------------------------


def make_q6_k_blob(
    N: int, K: int, seed: int = 0, uniform: bool = False
) -> torch.Tensor:
    """Build a ``(N, (K/256)*210)`` uint8 tensor of valid GGUF Q6_K blocks.

    With ``uniform=True`` every sub-block scale is identical so adjacent groups
    merge losslessly into a larger (MLX-supported) group size at export time.
    """
    assert K % QK_K == 0, f"K={K} must be a multiple of {QK_K}"
    nb = K // QK_K
    g = torch.Generator().manual_seed(seed)
    out = torch.empty(N, nb * Q6K_BLOCK_BYTES, dtype=torch.uint8)
    blocks = out.view(N, nb, Q6K_BLOCK_BYTES)
    # ql (0:128) + qh (128:192): any byte values are valid 6-bit quants.
    blocks[..., :192] = torch.randint(
        0, 256, (N, nb, 192), dtype=torch.uint8, generator=g
    )
    # scales (192:208): signed int8 scales (real Q6_K scales can be negative);
    # a modest magnitude keeps dequantized values sane.
    scales = torch.randint(-16, 17, (N, nb, 16), dtype=torch.int32, generator=g)
    blocks[..., 192:208] = scales.to(torch.int8).view(torch.uint8)
    # d (208:210): a small finite fp16 super-block scale. Chosen so dequantized
    # element magnitudes (~ d * scale * (q-32)) are O(0.1), like real Q6_K
    # weights -- the mat-mat kernel stores tiles in half precision (as in
    # llama.cpp), so unrealistically large magnitudes would exceed bf16 tol.
    blocks[..., 208:210] = torch.tensor([7e-4], dtype=torch.float16).view(torch.uint8)
    if uniform:
        blocks[..., 192:208] = 8  # identical int8 scale in every sub-block
    return out


def make_q4_k_blob(
    N: int, K: int, seed: int = 0, uniform: bool = False
) -> torch.Tensor:
    """Build a ``(N, (K/256)*144)`` uint8 tensor of valid GGUF Q4_K blocks.

    With ``uniform=True`` the packed sub-block scales/mins are identical so
    adjacent groups merge losslessly into a larger group size at export time.
    """
    assert K % QK_K == 0, f"K={K} must be a multiple of {QK_K}"
    nb = K // QK_K
    block_bytes = 144  # Q4_K: d(2) + dmin(2) + scales(12) + qs(128)
    g = torch.Generator().manual_seed(seed)
    out = torch.empty(N, nb * block_bytes, dtype=torch.uint8)
    blocks = out.view(N, nb, block_bytes)
    # d (0:2) / dmin (2:4): small finite fp16 super-block scale + min, so
    # dequantized magnitudes stay O(0.1) like real Q4_K weights.
    blocks[..., 0:2] = torch.tensor([7e-4], dtype=torch.float16).view(torch.uint8)
    blocks[..., 2:4] = torch.tensor([7e-4], dtype=torch.float16).view(torch.uint8)
    # scales+mins (4:16, 6-bit packed) and qs (16:144, 4-bit): any bytes valid.
    blocks[..., 4:144] = torch.randint(
        0, 256, (N, nb, 140), dtype=torch.uint8, generator=g
    )
    if uniform:
        blocks[..., 4:16] = 0x21  # identical packed 6-bit sub-block scales/mins
    return out


def make_q5_k_blob(
    N: int, K: int, seed: int = 0, uniform: bool = False
) -> torch.Tensor:
    """Build a ``(N, (K/256)*176)`` uint8 tensor of valid GGUF Q5_K blocks.

    With ``uniform=True`` the packed sub-block scales/mins are identical so
    adjacent groups merge losslessly into a larger group size at export time.
    """
    assert K % QK_K == 0, f"K={K} must be a multiple of {QK_K}"
    nb = K // QK_K
    block_bytes = Q5K_BLOCK_BYTES  # d(2)+dmin(2)+scales(12)+qh(32)+qs(128) = 176
    g = torch.Generator().manual_seed(seed)
    out = torch.empty(N, nb * block_bytes, dtype=torch.uint8)
    blocks = out.view(N, nb, block_bytes)
    # d (0:2) / dmin (2:4): small finite fp16 super-block scale + min, so
    # dequantized magnitudes stay O(0.1) like real Q5_K weights.
    blocks[..., 0:2] = torch.tensor([7e-4], dtype=torch.float16).view(torch.uint8)
    blocks[..., 2:4] = torch.tensor([7e-4], dtype=torch.float16).view(torch.uint8)
    # scales+mins (4:16, 6-bit packed), qh (16:48, high bit), qs (48:176, low
    # 4 bits): any bytes valid.
    blocks[..., 4:block_bytes] = torch.randint(
        0, 256, (N, nb, block_bytes - 4), dtype=torch.uint8, generator=g
    )
    if uniform:
        blocks[..., 4:16] = 0x21  # identical packed 6-bit sub-block scales/mins
    return out


_BLOB_MAKERS = {
    "q6_k": make_q6_k_blob,
    "q5_k": make_q5_k_blob,
    "q4_k": make_q4_k_blob,
}


@contextmanager
def _emit_direct_gguf_env(enabled: bool):
    old = os.environ.get("ET_MLX_EMIT_DIRECT_GGUF")
    os.environ["ET_MLX_EMIT_DIRECT_GGUF"] = "1" if enabled else "0"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("ET_MLX_EMIT_DIRECT_GGUF", None)
        else:
            os.environ["ET_MLX_EMIT_DIRECT_GGUF"] = old


def _make_gguf_linear_model(
    N: int,
    K: int,
    dtype: torch.dtype,
    bias: bool,
    ggml_type: str = "q6_k",
    seed: int = 0,
    uniform: bool = False,
) -> nn.Module:
    """An ``nn.Linear`` whose weight is a GGUF ``ExportableGGUFTensor``."""
    linear = nn.Linear(K, N, bias=bias).to(dtype)
    blob = _BLOB_MAKERS[ggml_type](N, K, seed=seed, uniform=uniform)
    linear.weight = nn.Parameter(
        ExportableGGUFTensor.from_raw(blob, ggml_type, dtype), requires_grad=False
    )
    return linear


class GGUFLinearModel(nn.Module):
    """Wrapper so the forward arg is named ``x`` (for dynamic-shape specs)."""

    def __init__(self, linear: nn.Module):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _fp32_linear_reference(model: "GGUFLinearModel", x: torch.Tensor):
    """fp32-accumulation reference matching the kernel.

    The kernels accumulate in fp32 and cast to the I/O dtype only at the end, so
    a bf16 eager matmul is too noisy an oracle over large K. Dequantize in fp32,
    matmul in fp32, then cast back -- differences collapse to ~1 output ULP.

    Direct Q6_K / Q4_K kernels dequantize the raw GGUF blob in-kernel; use the
    gguf-exact dequant as the reference oracle. Legacy Q4_K tests override this
    to match the export-time MLX qparam repack path.
    """
    lin = model.linear
    weight = lin.weight
    w = weight.dequantize(torch.float32)
    bias = lin.bias.float() if lin.bias is not None else None
    out = torch.nn.functional.linear(x.float(), w, bias)
    return [out.to(x.dtype)]


def _q4k_mlx_native_dequant(weight) -> torch.Tensor:
    from executorch.backends.mlx.builder.op_helpers import to_mlx_qparams

    intx = weight.to_intx_unpacked_to_int8_tensor()
    group_size = int(intx.block_size[-1])
    packed, biases = to_mlx_qparams(intx.qdata, intx.scale, intx.zero_point, 4)
    packed_bytes = packed.view(torch.uint8)
    nibbles = torch.stack(
        [(packed_bytes & 0xF).float(), ((packed_bytes >> 4) & 0xF).float()], dim=-1
    )
    q_unsigned = nibbles.reshape(intx.qdata.shape[0], -1)
    scale = intx.scale.float().repeat_interleave(group_size, dim=1)
    bias = biases.float().repeat_interleave(group_size, dim=1)
    return scale * q_unsigned + bias


_GGML_BITS = {"q4_k": 4, "q5_k": 5, "q6_k": 6}


def _mlx_native_dequant(weight) -> torch.Tensor:
    """Reference for the MLX-native repack path (q4_k / q5_k / q6_k).

    Mirrors the export-time repack (``to_intx_unpacked_to_int8_tensor`` with
    ``max_group_size=128``) and the MLX affine kernel, which computes
    ``scale*Q + bias == scale*(qdata - zero_point)``. Invariant to lossless
    group merging, so it validates the group-size upgrade too.
    """
    intx = weight.to_intx_unpacked_to_int8_tensor(max_group_size=128)
    gs = int(intx.block_size[-1])
    q = intx.qdata.float()
    scale = intx.scale.float().repeat_interleave(gs, dim=1)
    zp = intx.zero_point.float().repeat_interleave(gs, dim=1)
    return scale * (q - zp)


def _fp32_linear_mlx_native_reference(model: "GGUFLinearModel", x: torch.Tensor):
    lin = model.linear
    # q4_k keeps its packed-nibble oracle; q5_k/q6_k use the generic one.
    if lin.weight.ggml_type == "q4_k":
        w = _q4k_mlx_native_dequant(lin.weight)
    else:
        w = _mlx_native_dequant(lin.weight)
    bias = lin.bias.float() if lin.bias is not None else None
    out = torch.nn.functional.linear(x.float(), w, bias)
    return [out.to(x.dtype)]


_DTYPE_TOL = {
    torch.bfloat16: (2e-2, 2e-2),
    # The mat-mat (prefill) kernel stores tiles in half precision (as in
    # llama.cpp), so fp16 outputs are accurate to ~half precision (~4e-3).
    torch.float16: (5e-3, 5e-3),
    torch.float32: (1e-4, 1e-4),
}
_DTYPE_TAG = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}


def _edge_compile_config():
    from executorch.exir import EdgeCompileConfig

    # The dequantize_gguf custom op isn't a core ATen op; skip IR validity.
    return EdgeCompileConfig(_check_ir_validity=False)


class GGUFLinearTest(OpTestCase):
    name = "gguf_linear"

    def __init__(
        self,
        M: int = 1,
        N: int = 256,
        K: int = 256,
        dtype: torch.dtype = torch.bfloat16,
        bias: bool = True,
        ggml_type: str = "q6_k",
        emit_direct_gguf: bool = True,
        uniform: bool = False,
    ):
        self.M = M
        self.N = N
        self.K = K
        self.dtype = dtype
        self.bias = bias
        self.ggml_type = ggml_type
        self.emit_direct_gguf = emit_direct_gguf
        self.uniform = uniform
        self.rtol, self.atol = _DTYPE_TOL[dtype]
        if ggml_type == "q5_k" and dtype == torch.float16:
            self.rtol, self.atol = 2e-2, 2e-2
        tag = f"gguf_linear_{ggml_type}_m{M}_n{N}_k{K}_{_DTYPE_TAG[dtype]}"
        if not emit_direct_gguf:
            tag += "_mlx_native"
        if uniform:
            tag += "_merged"
        self.name = tag if bias else tag + "_nobias"

    @classmethod
    def get_test_configs(cls) -> List["GGUFLinearTest"]:
        cfgs: List["GGUFLinearTest"] = []
        # Decode (mat-vec).
        for K in (256, 512, 1024):
            for N in (256, 512):
                cfgs.append(cls(M=1, N=N, K=K, dtype=torch.bfloat16))
        cfgs.append(cls(M=1, N=256, K=256, dtype=torch.float16))
        cfgs.append(cls(M=1, N=256, K=256, dtype=torch.float32))
        cfgs.append(cls(M=1, N=256, K=256, dtype=torch.bfloat16, bias=False))
        # Prefill (mat-mat).
        for M in (8, 64, 128):
            cfgs.append(cls(M=M, N=512, K=512, dtype=torch.bfloat16))
        cfgs.append(cls(M=32, N=256, K=256, dtype=torch.float16))
        # Ragged shapes (M and N not multiples of the 32-wide tile / row group).
        cfgs.append(cls(M=40, N=300, K=256, dtype=torch.bfloat16))
        cfgs.append(cls(M=1, N=300, K=256, dtype=torch.bfloat16))
        # Real Gemma-4-31B shapes (hidden=5376, ffn=21504) at production N/K.
        cfgs.append(cls(M=1, N=4096, K=5376, dtype=torch.bfloat16))  # attn_v
        cfgs.append(cls(M=1, N=5376, K=21504, dtype=torch.bfloat16))  # ffn_down
        cfgs.append(cls(M=8, N=5376, K=21504, dtype=torch.bfloat16))  # ffn_down prefill
        # lm_head: real vocab is 262144, but N is capped so the packed weight
        # fits CI-runner GPU buffer limits; the mat-vec N-tiling path is the
        # same at any N.
        cfgs.append(cls(M=1, N=16384, K=5376, dtype=torch.bfloat16))  # lm_head
        # Q4_K: exercise both the fused direct path and the legacy MLX-native
        # repack path on each shape. Each path uses its own reference oracle
        # (see compute_expected_outputs).
        q4k_shapes = [
            {"M": 1, "N": 512, "K": 512, "dtype": torch.bfloat16},
            {"M": 8, "N": 512, "K": 512, "dtype": torch.bfloat16},
            {"M": 1, "N": 5376, "K": 5376, "dtype": torch.bfloat16},
            {"M": 1, "N": 512, "K": 512, "dtype": torch.bfloat16, "bias": False},
        ]
        for shape in q4k_shapes:
            for emit_direct in (True, False):
                cfgs.append(
                    cls(ggml_type="q4_k", emit_direct_gguf=emit_direct, **shape)
                )
        # Q5_K: fused direct path only (no MLX-native repack). Decode (mat-vec),
        # prefill (mat-mat), ragged M/N, plus a production-size shape.
        cfgs.append(cls(ggml_type="q5_k", M=1, N=256, K=256, dtype=torch.bfloat16))
        cfgs.append(cls(ggml_type="q5_k", M=1, N=512, K=1024, dtype=torch.bfloat16))
        cfgs.append(cls(ggml_type="q5_k", M=1, N=256, K=256, dtype=torch.float16))
        cfgs.append(cls(ggml_type="q5_k", M=1, N=256, K=256, dtype=torch.float32))
        cfgs.append(
            cls(ggml_type="q5_k", M=1, N=256, K=256, dtype=torch.bfloat16, bias=False)
        )
        cfgs.append(cls(ggml_type="q5_k", M=8, N=512, K=512, dtype=torch.bfloat16))
        cfgs.append(cls(ggml_type="q5_k", M=128, N=512, K=512, dtype=torch.bfloat16))
        cfgs.append(cls(ggml_type="q5_k", M=32, N=256, K=256, dtype=torch.float16))
        cfgs.append(cls(ggml_type="q5_k", M=40, N=300, K=256, dtype=torch.bfloat16))
        cfgs.append(cls(ggml_type="q5_k", M=1, N=300, K=256, dtype=torch.bfloat16))
        cfgs.append(cls(ggml_type="q5_k", M=1, N=5376, K=5376, dtype=torch.bfloat16))
        # Q5_K MLX-native repack path (bits=5 affine QuantizedMatmul), decode +
        # prefill + nobias.
        cfgs.append(cls(ggml_type="q5_k", emit_direct_gguf=False, M=1, N=512, K=512))
        cfgs.append(cls(ggml_type="q5_k", emit_direct_gguf=False, M=8, N=512, K=512))
        cfgs.append(
            cls(
                ggml_type="q5_k",
                emit_direct_gguf=False,
                M=1,
                N=512,
                K=512,
                bias=False,
            )
        )
        # Group-size upgrade: uniform sub-blocks merge to group_size 128, so the
        # affine kernels run at gs>32 (and q6_k's native path fires at all).
        for gt in ("q4_k", "q5_k", "q6_k"):
            cfgs.append(
                cls(
                    ggml_type=gt,
                    emit_direct_gguf=False,
                    uniform=True,
                    M=1,
                    N=512,
                    K=512,
                )
            )
        cfgs.append(
            cls(
                ggml_type="q6_k",
                emit_direct_gguf=False,
                uniform=True,
                M=8,
                N=512,
                K=512,
            )
        )
        return cfgs

    def generate_test_files(self, verbose: bool = False):
        with _emit_direct_gguf_env(self.emit_direct_gguf):
            return super().generate_test_files(verbose=verbose)

    def get_edge_compile_config(self):
        return _edge_compile_config()

    def create_model(self) -> nn.Module:
        return GGUFLinearModel(
            _make_gguf_linear_model(
                self.N,
                self.K,
                self.dtype,
                self.bias,
                self.ggml_type,
                uniform=self.uniform,
            )
        )

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        torch.manual_seed(0)
        return (torch.randn(self.M, self.K, dtype=self.dtype),)

    def _native_fires(self) -> bool:
        # MLX-native repack runs when direct emission is off and the weight
        # merges to an MLX group size: q4_k/q5_k always (native 32); q6_k
        # (native 16) only when merging reaches >= 32 (uniform fixtures).
        if self.emit_direct_gguf:
            return False
        if self.ggml_type in ("q4_k", "q5_k"):
            return True
        return self.uniform

    def compute_expected_outputs(self, model, test_inputs):
        if self._native_fires():
            return _fp32_linear_mlx_native_reference(model, test_inputs[0])
        return _fp32_linear_reference(model, test_inputs[0])


class GGUFLinearDynamicTest(OpTestCase):
    """Dynamic seqlen: export once with a symbolic M, run with M=1 (decode /
    else chain) and M>1 (prefill / then chain) to exercise both IfNode branches.
    """

    name = "gguf_linear_dynamic"

    def __init__(
        self,
        export_M: int = 4,
        test_M: int = 1,
        N: int = 512,
        K: int = 512,
        dtype: torch.dtype = torch.bfloat16,
        ggml_type: str = "q6_k",
        emit_direct_gguf: bool = True,
    ):
        self.export_M = export_M
        self.test_M = test_M
        self.N = N
        self.K = K
        self.dtype = dtype
        self.ggml_type = ggml_type
        self.emit_direct_gguf = emit_direct_gguf
        self.rtol, self.atol = _DTYPE_TOL[dtype]
        name = (
            f"gguf_linear_dyn_{ggml_type}_exp{export_M}_test{test_M}_n{N}_k{K}_"
            f"{_DTYPE_TAG[dtype]}"
        )
        if not emit_direct_gguf:
            name += "_mlx_native"
        self.name = name

    @classmethod
    def get_test_configs(cls) -> List["GGUFLinearDynamicTest"]:
        cfgs = [
            cls(export_M=4, test_M=1, dtype=torch.bfloat16),  # decode / else
            cls(export_M=4, test_M=8, dtype=torch.bfloat16),  # prefill / then
            cls(export_M=4, test_M=4, dtype=torch.bfloat16),  # control
            cls(export_M=4, test_M=1, dtype=torch.float16),
            cls(export_M=4, test_M=40, N=300, K=256, dtype=torch.bfloat16),  # ragged
        ]
        # Q4_K: exercise both the fused direct path and the legacy MLX-native
        # repack path on each shape. Each path needs its own reference oracle
        # (see compute_expected_outputs).
        for emit_direct in (True, False):
            cfgs.append(
                cls(
                    export_M=4,
                    test_M=1,
                    dtype=torch.bfloat16,
                    ggml_type="q4_k",
                    emit_direct_gguf=emit_direct,
                )
            )
            cfgs.append(
                cls(
                    export_M=4,
                    test_M=8,
                    dtype=torch.bfloat16,
                    ggml_type="q4_k",
                    emit_direct_gguf=emit_direct,
                )
            )
        # Q5_K dynamic: exercise both IfNode branches (decode mat-vec / prefill
        # mat-mat) from a single symbolic-M export.
        cfgs.append(cls(export_M=4, test_M=1, dtype=torch.bfloat16, ggml_type="q5_k"))
        cfgs.append(cls(export_M=4, test_M=8, dtype=torch.bfloat16, ggml_type="q5_k"))
        # Q5_K dynamic MLX-native repack (single QuantizedMatmul serves both M).
        cfgs.append(
            cls(
                export_M=4,
                test_M=1,
                dtype=torch.bfloat16,
                ggml_type="q5_k",
                emit_direct_gguf=False,
            )
        )
        cfgs.append(
            cls(
                export_M=4,
                test_M=8,
                dtype=torch.bfloat16,
                ggml_type="q5_k",
                emit_direct_gguf=False,
            )
        )
        return cfgs

    def get_dynamic_shapes(self):
        seq_dim = torch.export.Dim("seq_len", min=1, max=64)
        return {"x": {0: seq_dim}}

    def generate_test_files(self, verbose: bool = False):
        with _emit_direct_gguf_env(self.emit_direct_gguf):
            return super().generate_test_files(verbose=verbose)

    def get_edge_compile_config(self):
        return _edge_compile_config()

    def create_model(self) -> nn.Module:
        # Deterministic weight so export-time and run-time use the same model.
        return GGUFLinearModel(
            _make_gguf_linear_model(
                self.N, self.K, self.dtype, bias=True, ggml_type=self.ggml_type
            )
        )

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        torch.manual_seed(0)
        return (torch.randn(self.export_M, self.K, dtype=self.dtype),)

    def create_test_inputs(self) -> Tuple[torch.Tensor, ...]:
        torch.manual_seed(0)
        return (torch.randn(self.test_M, self.K, dtype=self.dtype),)

    def compute_expected_outputs(self, model, test_inputs):
        # Dynamic configs only use the native path for q4_k/q5_k, which always
        # merges to an MLX group size, so "direct off" implies the native path.
        if not self.emit_direct_gguf:
            return _fp32_linear_mlx_native_reference(model, test_inputs[0])
        return _fp32_linear_reference(model, test_inputs[0])


def _eager_sanity() -> None:
    """Quick CPU check: the subclass linear exports to dequantize_gguf."""
    model = GGUFLinearModel(_make_gguf_linear_model(4, 512, torch.bfloat16, bias=True))
    x = torch.randn(3, 512, dtype=torch.bfloat16)
    out = model(x)
    print(
        f"eager forward finite: {torch.isfinite(out).all().item()}, shape {tuple(out.shape)}"
    )
    ep = torch.export.export(model, (x,)).run_decompositions({})
    targets = {str(n.target) for n in ep.graph.nodes if n.op == "call_function"}
    assert "torchao.dequantize_gguf.default" in targets, targets
    print("export contains torchao.dequantize_gguf: OK")


if __name__ == "__main__":  # noqa: C901
    import argparse
    import sys

    from executorch.backends.mlx.test.test_utils import rebuild_op_test_runner

    parser = argparse.ArgumentParser(
        description="Test GGUF Q4_K / Q5_K / Q6_K linear lowering"
    )
    parser.add_argument(
        "action", choices=["generate", "compare", "run", "list", "eager"]
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    if args.action == "eager":
        _eager_sanity()
        sys.exit(0)

    if args.rebuild and not rebuild_op_test_runner(verbose=args.verbose):
        sys.exit(1)

    configs = (
        GGUFLinearTest.get_test_configs() + GGUFLinearDynamicTest.get_test_configs()
    )

    if args.action == "list":
        for cfg in configs:
            print(f"  {cfg.name}")
        sys.exit(0)

    if args.config:
        configs = [c for c in configs if c.name == args.config]
        if not configs:
            print(f"No config matching '{args.config}'")
            sys.exit(1)

    passed = 0
    failed = 0
    failed_names: List[str] = []

    for test in configs:
        if args.action == "generate":
            pte_path, _, _ = test.generate_test_files(verbose=args.verbose)
            print(f"Generated: {pte_path}")
        elif args.action == "compare":
            actual_path = test.get_test_dir() / "actual_output.bin"
            ok, msg = test.compare_with_actual(actual_path)
            print(f"{'✓' if ok else '✗'} {test.name}: {msg}")
            if ok:
                passed += 1
            else:
                failed += 1
                failed_names.append(test.name)
        elif args.action == "run":
            ok = test.run_test(verbose=args.verbose)
            if ok:
                passed += 1
            else:
                failed += 1
                failed_names.append(test.name)

    if args.action in ("run", "compare"):
        print(f"\nPassed: {passed}, Failed: {failed}")
        if failed_names:
            print(f"Failed: {', '.join(failed_names)}")
        sys.exit(0 if failed == 0 else 1)
