# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F
import triton
from executorch.backends.cuda.dp4a_planar_int5_tensor import pack_int5
from executorch.backends.cuda.dp4a_planar_int6_tensor import pack_int6
from executorch.backends.cuda.triton.kernels.int4_quantized_gemm import (
    INT4_QUANTIZED_GEMM,
)
from executorch.backends.cuda.triton.kernels.q8_mmq import (
    _prepare_mmq_weight,
    _quantize_mmq_activation,
    launch_q8_mmq,
)
from torch.library import wrap_triton

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("group_size", [16, 32, 128, 256])
def test_activation_rounding_and_zero_groups(group_size):
    """Check quantized values, metadata and the zero-block convention exactly."""
    torch.manual_seed(7)
    x = torch.randn(9, 512, dtype=torch.bfloat16, device="cuda")
    x[0].zero_()
    grouped = x.float().reshape(9, -1, group_size)
    expected_scale = grouped.abs().amax(-1) / 127
    expected_scale = torch.where(expected_scale > 0, expected_scale, 1.0)
    expected = (grouped / expected_scale.unsqueeze(-1)).round().to(torch.int8)
    q = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty_like(expected_scale)
    sums = torch.empty_like(expected_scale, dtype=torch.int32)
    wrap_triton(_quantize_mmq_activation)[
        (triton.cdiv(9, 8), triton.cdiv(512 // group_size, 4))
    ](x, q, scale, sums, 9, 512, group_size, 8, 4)
    torch.testing.assert_close(q.reshape_as(expected), expected, rtol=0, atol=0)
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(
        sums, expected.to(torch.int32).sum(-1, dtype=torch.int32), rtol=0, atol=0
    )


@pytest.mark.parametrize("bits", [4, 5, 6])
@pytest.mark.parametrize("m", [65, 257, 4096])
def test_mmq_matches_independent_groupwise_reference(bits, m):
    """Compare integer dots against an independent floating point oracle."""
    torch.manual_seed(19)
    torch.backends.cuda.matmul.allow_tf32 = False
    n, k = 129, 256
    gs = 16 if bits == 6 else 32
    q = torch.randint(
        -32 if bits == 6 else 0,
        16 if bits == 4 else 32,
        (n, k),
        device="cuda",
        dtype=torch.int8,
    )
    if bits == 4:
        ql = q[:, 0::2].to(torch.uint8) | (q[:, 1::2].to(torch.uint8) << 4)
        qh = ql
    elif bits == 5:
        ql, qh = pack_int5(q.to(torch.uint8))
    else:
        ql, qh = pack_int6(q)
    codes = torch.randint(
        -63 if bits == 6 else 1,
        64,
        (n, k // gs),
        dtype=torch.int8 if bits == 6 else torch.uint8,
        device="cuda",
    )
    steps = torch.full((n, k // 256), 0.003, dtype=torch.float16, device="cuda")
    zeros = torch.full(
        (n, k // gs), 8 if bits == 4 else 16, dtype=torch.uint8, device="cuda"
    )
    zsteps = (0.5 + torch.rand_like(steps)).to(torch.float16)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    out = launch_q8_mmq(x, ql, qh, codes, steps, zeros, zsteps, bits=bits)
    grouped = x.float().reshape(m, k // 256, 256)
    scale = grouped.abs().amax(-1, keepdim=True) / 127
    scale = torch.where(scale > 0, scale, 1.0)
    activation = (grouped / scale).round().clamp(-127, 127) * scale
    weight_scale = (
        codes.float() * steps.float().repeat_interleave(256 // gs, 1)
    ).float()
    zero = (
        0
        if bits == 6
        else (zeros.float() * zsteps.float().repeat_interleave(256 // gs, 1))
        .float()
        .unsqueeze(-1)
    )
    weight = (q.float().reshape(n, k // gs, gs) - zero) * weight_scale.unsqueeze(-1)
    ref = F.linear(activation.reshape(m, k), weight.reshape(n, k)).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, rtol=0.009, atol=0.04)


def test_dynamic_large_bucket_compiles():
    """Keep K static while exporting and compiling bounded dynamic rows."""

    class Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter(
                "q",
                torch.nn.Parameter(
                    torch.full((33, 128), 137, dtype=torch.uint8, device="cuda"),
                    requires_grad=False,
                ),
            )
            self.register_parameter(
                "s",
                torch.nn.Parameter(
                    torch.ones((33, 8), dtype=torch.uint8, device="cuda"),
                    requires_grad=False,
                ),
            )
            self.register_parameter(
                "step",
                torch.nn.Parameter(
                    torch.full((33, 1), 0.01, dtype=torch.float16, device="cuda"),
                    requires_grad=False,
                ),
            )
            self.register_parameter(
                "z",
                torch.nn.Parameter(
                    torch.full((33, 8), 8, dtype=torch.uint8, device="cuda"),
                    requires_grad=False,
                ),
            )
            self.register_parameter(
                "zstep",
                torch.nn.Parameter(
                    torch.ones((33, 1), dtype=torch.float16, device="cuda"),
                    requires_grad=False,
                ),
            )

        def forward(self, x):
            return INT4_QUANTIZED_GEMM.op(4096)(
                x.reshape(-1, 256), self.q, self.s, self.step, self.z, self.zstep, 32
            )

    model = Linear()
    x = torch.randn(65, 256, dtype=torch.bfloat16, device="cuda")
    ep = torch.export.export(
        model, (x,), dynamic_shapes=({0: torch.export.Dim("m", min=5, max=4096)},)
    )
    exported = ep.module()
    for buffer in exported.buffers():
        torch._dynamo.mark_static(buffer)
    compiled = torch.compile(exported, fullgraph=True, dynamic=True)
    for m in [5, 31, 65, 129, 4096]:
        value = torch.randn(m, 256, dtype=torch.bfloat16, device="cuda")
        torch.testing.assert_close(compiled(value), model(value), rtol=0.009, atol=0.04)


def test_q4_large_zero_center_does_not_wrap():
    """Large zero metadata must not overflow the two-byte integer product."""
    m, n, k = 65, 33, 256
    x = torch.ones(m, k, dtype=torch.bfloat16, device="cuda")
    q = torch.full((n, k // 2), 0x11, dtype=torch.uint8, device="cuda")
    scale = torch.full((n, k // 32), 255, dtype=torch.uint8, device="cuda")
    step = torch.full((n, k // 256), 0.003, dtype=torch.float16, device="cuda")
    zero = torch.full_like(scale, 255)
    zstep = torch.full_like(step, 64)
    out = launch_q8_mmq(x, q, q, scale, step, zero, zstep, bits=4)
    weight = (
        (1 - zero.float() * zstep.float()) * scale.float() * step.float()
    ).repeat_interleave(32, dim=1)
    ref = F.linear(x.float(), weight).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, rtol=0.009, atol=0.04)


def test_prepared_residual_preserves_fma_rounding():
    """Use FP64 to check one FP32 rounding before residuals become BF16."""
    torch.manual_seed(73)
    n, k = 129, 512
    q = torch.zeros((n, k // 2), dtype=torch.uint8, device="cuda")
    codes = torch.randint(1, 256, (n, k // 32), dtype=torch.uint8, device="cuda")
    zeros = torch.randint(1, 256, (n, k // 32), dtype=torch.uint8, device="cuda")
    steps = torch.full((n, k // 256), 0.01, dtype=torch.float16, device="cuda")
    zsteps = torch.full_like(steps, 0.00313)
    lo = torch.empty((n, k), dtype=torch.int8, device="cuda")
    hi = torch.empty_like(lo)
    residual = torch.empty((n, k // 32), dtype=torch.bfloat16, device="cuda")
    wrap_triton(_prepare_mmq_weight)[(triton.cdiv(n, 4), k // 256)](
        q, q, codes, steps, zeros, zsteps, lo, hi, residual, n, k, 4, 4
    )
    product = (codes.int() * zeros.int()).float()
    dz = zsteps.float().repeat_interleave(8, 1)
    integer = (product * dz).round().clamp(-8192, 8192)
    expected = (
        (integer.double() - product.double() * dz.double()).float().to(torch.bfloat16)
    )
    unfused = (integer - product * dz).to(torch.bfloat16)
    assert not torch.equal(
        expected, unfused
    ), "Inputs must distinguish the two rounding rules"
    torch.testing.assert_close(residual, expected, rtol=0, atol=0)
