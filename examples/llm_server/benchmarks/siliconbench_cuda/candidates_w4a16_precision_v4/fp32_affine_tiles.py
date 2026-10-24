# Optimized by KernelAgent-Oink(https://github.com/meta-pytorch/KernelAgent)
"""Precision-v4 FP32-affine W4A16 low-stage and split8 schedules.

Packed qdata/codes/FP16 steps are unchanged. The mainloop is the validated
precision-v3 K256 implementation; v4 adds explicit M16 stage1/2-warp/split8
bases while retaining the measured M8 SM120 controls.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

from . import common

CONFIGS: dict[str, dict[str, int]] = {
    "precision_v4_m8_n32_w2_s2_sp1": {"bucket": 8, "block_n": 32, "num_warps": 2, "num_stages": 2, "split_k": 1},
    "precision_v4_m8_n32_w2_s2_sp8": {"bucket": 8, "block_n": 32, "num_warps": 2, "num_stages": 2, "split_k": 8},
    "precision_v4_m16_n64_w4_s1_sp1": {"bucket": 16, "block_n": 64, "num_warps": 4, "num_stages": 1, "split_k": 1},
    "precision_v4_m16_n64_w8_s1_sp1": {"bucket": 16, "block_n": 64, "num_warps": 8, "num_stages": 1, "split_k": 1},
    "precision_v4_m16_n64_w2_s1_sp1": {"bucket": 16, "block_n": 64, "num_warps": 2, "num_stages": 1, "split_k": 1},
    "precision_v4_m16_n32_w2_s1_sp1": {"bucket": 16, "block_n": 32, "num_warps": 2, "num_stages": 1, "split_k": 1},
    "precision_v4_m16_n64_w8_s1_sp8": {"bucket": 16, "block_n": 64, "num_warps": 8, "num_stages": 1, "split_k": 8},
    "precision_v4_m16_n64_w4_s1_sp8": {"bucket": 16, "block_n": 64, "num_warps": 4, "num_stages": 1, "split_k": 8},
    "precision_v4_m16_n64_w2_s1_sp8": {"bucket": 16, "block_n": 64, "num_warps": 2, "num_stages": 1, "split_k": 8},
    "precision_v4_m16_n32_w4_s1_sp8": {"bucket": 16, "block_n": 32, "num_warps": 4, "num_stages": 1, "split_k": 8},
    "precision_v4_m16_n32_w2_s1_sp8": {"bucket": 16, "block_n": 32, "num_warps": 2, "num_stages": 1, "split_k": 8},
    "precision_v4_m16_n64_w4_s2_sp8": {"bucket": 16, "block_n": 64, "num_warps": 4, "num_stages": 2, "split_k": 8},
}
_ALLOWED_CONFIG_KEYS = {"bucket", "block_n", "num_warps", "num_stages", "split_k"}


@triton.jit
def _decode_fp32_affine_superblock(
    qdata, scale, scale_step, zero, zero_step,
    offs_n, n_mask, super_block,
    stride_qn: tl.constexpr, stride_qk: tl.constexpr,
    stride_sn: tl.constexpr, stride_sk: tl.constexpr,
    stride_ssn: tl.constexpr, stride_ssk: tl.constexpr,
    stride_zn: tl.constexpr, stride_zk: tl.constexpr,
    stride_zsn: tl.constexpr, stride_zsk: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Reconstruct FP32 affine weights and round only the final MMA operand."""
    offs_packed = tl.arange(0, 128)
    packed = tl.load(
        qdata + offs_n[:, None] * stride_qn
        + (super_block * 128 + offs_packed[None, :]) * stride_qk,
        mask=n_mask[:, None], other=0,
    ).to(tl.uint32)
    q = tl.interleave(packed & 0xF, (packed >> 4) & 0xF).to(tl.float32)
    offs_group = tl.arange(0, 8)
    group = super_block * 8 + offs_group
    scale_code = tl.load(
        scale + offs_n[:, None] * stride_sn + group[None, :] * stride_sk,
        mask=n_mask[:, None], other=0,
    ).to(tl.float32)
    zero_code = tl.load(
        zero + offs_n[:, None] * stride_zn + group[None, :] * stride_zk,
        mask=n_mask[:, None], other=0,
    ).to(tl.float32)
    scale_step_value = tl.load(
        scale_step + offs_n * stride_ssn + super_block * stride_ssk,
        mask=n_mask, other=0.0,
    ).to(tl.float32)
    zero_step_value = tl.load(
        zero_step + offs_n * stride_zsn + super_block * stride_zsk,
        mask=n_mask, other=0.0,
    ).to(tl.float32)
    scale_group = scale_code * scale_step_value[:, None]
    zero_group = zero_code * zero_step_value[:, None]
    scale_expanded = tl.reshape(
        tl.broadcast_to(scale_group[:, :, None], (BLOCK_N, 8, 32)),
        (BLOCK_N, 256), can_reorder=False,
    )
    zero_expanded = tl.reshape(
        tl.broadcast_to(zero_group[:, :, None], (BLOCK_N, 8, 32)),
        (BLOCK_N, 256), can_reorder=False,
    )
    return ((q - zero_expanded) * scale_expanded).to(tl.bfloat16)


@triton.jit
def _precision_direct_kernel(
    x, qdata, scale, scale_step, zero, zero_step, out, M,
    N: tl.constexpr, K: tl.constexpr,
    stride_xm: tl.constexpr, stride_xk: tl.constexpr,
    stride_qn: tl.constexpr, stride_qk: tl.constexpr,
    stride_sn: tl.constexpr, stride_sk: tl.constexpr,
    stride_ssn: tl.constexpr, stride_ssk: tl.constexpr,
    stride_zn: tl.constexpr, stride_zk: tl.constexpr,
    stride_zsn: tl.constexpr, stride_zsk: tl.constexpr,
    stride_om: tl.constexpr, stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, PIPELINE_STAGES: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N
    offs_k = tl.arange(0, 256)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for super_block in tl.range(0, K // 256, num_stages=PIPELINE_STAGES):
        weight = _decode_fp32_affine_superblock(
            qdata, scale, scale_step, zero, zero_step, offs_n, n_mask, super_block,
            stride_qn, stride_qk, stride_sn, stride_sk, stride_ssn, stride_ssk,
            stride_zn, stride_zk, stride_zsn, stride_zsk, BLOCK_N,
        )
        activation = tl.load(
            x + offs_m[:, None] * stride_xm + (super_block * 256 + offs_k[None, :]) * stride_xk,
            mask=m_mask[:, None], other=0.0,
        ).to(tl.bfloat16)
        acc += tl.dot(activation, tl.trans(weight), out_dtype=tl.float32)
    tl.store(
        out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _precision_splitk_kernel(
    x, qdata, scale, scale_step, zero, zero_step, out, M,
    N: tl.constexpr, K: tl.constexpr,
    stride_xm: tl.constexpr, stride_xk: tl.constexpr,
    stride_qn: tl.constexpr, stride_qk: tl.constexpr,
    stride_sn: tl.constexpr, stride_sk: tl.constexpr,
    stride_ssn: tl.constexpr, stride_ssk: tl.constexpr,
    stride_zn: tl.constexpr, stride_zk: tl.constexpr,
    stride_zsn: tl.constexpr, stride_zsk: tl.constexpr,
    stride_os: tl.constexpr, stride_om: tl.constexpr, stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr, PIPELINE_STAGES: tl.constexpr,
):
    pid_n = tl.program_id(0)
    split_id = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N
    offs_k = tl.arange(0, 256)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    num_supers: tl.constexpr = K // 256
    base: tl.constexpr = num_supers // SPLIT_K
    extra: tl.constexpr = num_supers % SPLIT_K
    first_super = split_id * base + tl.minimum(split_id, extra)
    count = base + (split_id < extra)
    for super_block in tl.range(first_super, first_super + count, num_stages=PIPELINE_STAGES):
        weight = _decode_fp32_affine_superblock(
            qdata, scale, scale_step, zero, zero_step, offs_n, n_mask, super_block,
            stride_qn, stride_qk, stride_sn, stride_sk, stride_ssn, stride_ssk,
            stride_zn, stride_zk, stride_zsn, stride_zsk, BLOCK_N,
        )
        activation = tl.load(
            x + offs_m[:, None] * stride_xm + (super_block * 256 + offs_k[None, :]) * stride_xk,
            mask=m_mask[:, None], other=0.0,
        ).to(tl.bfloat16)
        acc += tl.dot(activation, tl.trans(weight), out_dtype=tl.float32)
    tl.store(
        out + split_id * stride_os + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc, mask=m_mask[:, None] & n_mask[None, :],
    )


def _default_config(bucket: int) -> dict[str, int]:
    cid = "precision_v4_m8_n32_w2_s2_sp1" if bucket == 8 else "precision_v4_m16_n64_w4_s1_sp1"
    return dict(CONFIGS[cid])


def _resolve_config(config: str | dict[str, Any] | None, bucket: int) -> dict[str, int]:
    if isinstance(config, str):
        if config not in CONFIGS:
            raise ValueError(f"unknown config id {config!r}")
        result = dict(CONFIGS[config])
    else:
        result = common.normalize_config(config, _default_config(bucket), _ALLOWED_CONFIG_KEYS)
    if result["bucket"] != bucket:
        raise ValueError("config bucket mismatch")
    if result["block_n"] not in (32, 64) or result["num_warps"] not in (2, 4, 8):
        raise ValueError("unsupported bounded block_n/warp schedule")
    if result["num_stages"] not in (1, 2, 3):
        raise ValueError("num_stages must be 1, 2, or 3")
    base = {key: result[key] for key in ("bucket", "block_n", "num_warps", "num_stages")}
    if not any(all(candidate[key] == value for key, value in base.items()) for candidate in CONFIGS.values()):
        raise ValueError(f"schedule is outside the bounded manifest: {base}")
    return result


def run(x, qdata, scale, scale_step, zero, zero_step, *, bucket: int, config=None) -> torch.Tensor:
    problem = common.validate_inputs(x, qdata, scale, scale_step, zero, zero_step, bucket=bucket)
    chosen = _resolve_config(config, bucket)
    split_k = chosen["split_k"]
    common.validate_split(split_k, problem.k)
    out, destination, split_stride = common.allocate_output(problem, split_k)
    strides = common.weight_strides(qdata, scale, scale_step, zero, zero_step)
    kernel = _precision_direct_kernel if split_k == 1 else _precision_splitk_kernel
    grid = (
        (triton.cdiv(problem.n, chosen["block_n"]),)
        if split_k == 1 else (triton.cdiv(problem.n, chosen["block_n"]), split_k)
    )
    launch = dict(
        N=problem.n, K=problem.k,
        stride_xm=x.stride(0), stride_xk=x.stride(1),
        stride_qn=strides[0], stride_qk=strides[1],
        stride_sn=strides[2], stride_sk=strides[3],
        stride_ssn=strides[4], stride_ssk=strides[5],
        stride_zn=strides[6], stride_zk=strides[7],
        stride_zsn=strides[8], stride_zsk=strides[9],
        stride_om=destination.stride(-2), stride_on=destination.stride(-1),
        BLOCK_M=bucket, BLOCK_N=chosen["block_n"], PIPELINE_STAGES=chosen["num_stages"],
        num_warps=chosen["num_warps"], num_stages=chosen["num_stages"],
    )
    if split_k > 1:
        launch.update(stride_os=split_stride, SPLIT_K=split_k)
    kernel[grid](x, qdata, scale, scale_step, zero, zero_step, destination, problem.m, **launch)
    common.finish_split(destination, out, split_k)
    return out


DIRECT_KERNEL = _precision_direct_kernel
SPLIT_KERNEL = _precision_splitk_kernel

__all__ = ["CONFIGS", "DIRECT_KERNEL", "SPLIT_KERNEL", "run"]
