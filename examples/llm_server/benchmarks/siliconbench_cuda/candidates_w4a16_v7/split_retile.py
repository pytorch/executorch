# Optimized by KernelAgent-Oink(https://github.com/meta-pytorch/KernelAgent)
"""Feedback-led W4A16 split-count and deterministic-reducer retiling.

The K256 FP32-affine/final-BF16 producer is unchanged from validated v5.
Only split count and reducer tile/warp schedules vary.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

from . import common

CONFIGS: dict[str, dict[str, int]] = {
    "v7_m8_n32_w2_s2_sp1": {"bucket": 8, "block_m": 8, "block_n": 32, "num_warps": 2, "num_stages": 2, "split_k": 1, "reduce_block_n": 64, "reduce_warps": 4},
    "v7_m8_n32_w2_s2_sp2_r64w4": {"bucket": 8, "block_m": 8, "block_n": 32, "num_warps": 2, "num_stages": 2, "split_k": 2, "reduce_block_n": 64, "reduce_warps": 4},
    "v7_m8_n32_w2_s2_sp4_r64w4": {"bucket": 8, "block_m": 8, "block_n": 32, "num_warps": 2, "num_stages": 2, "split_k": 4, "reduce_block_n": 64, "reduce_warps": 4},
    "v7_m8_n32_w2_s2_sp8_r64w4": {"bucket": 8, "block_m": 8, "block_n": 32, "num_warps": 2, "num_stages": 2, "split_k": 8, "reduce_block_n": 64, "reduce_warps": 4},
    "v7_m8_n32_w2_s2_sp8_r32w2": {"bucket": 8, "block_m": 8, "block_n": 32, "num_warps": 2, "num_stages": 2, "split_k": 8, "reduce_block_n": 32, "reduce_warps": 2},
    "v7_m8_n32_w2_s2_sp8_r128w8": {"bucket": 8, "block_m": 8, "block_n": 32, "num_warps": 2, "num_stages": 2, "split_k": 8, "reduce_block_n": 128, "reduce_warps": 8},
    "v7_m16_n64_w8_s1_sp1": {"bucket": 16, "block_m": 16, "block_n": 64, "num_warps": 8, "num_stages": 1, "split_k": 1, "reduce_block_n": 64, "reduce_warps": 4},
    "v7_m16_n64_w4_s2_sp1": {"bucket": 16, "block_m": 16, "block_n": 64, "num_warps": 4, "num_stages": 2, "split_k": 1, "reduce_block_n": 64, "reduce_warps": 4},
    "v7_m16_n64_w8_s1_sp4_r64w4": {"bucket": 16, "block_m": 16, "block_n": 64, "num_warps": 8, "num_stages": 1, "split_k": 4, "reduce_block_n": 64, "reduce_warps": 4},
    "v7_m16_n64_w8_s1_sp8_r64w4": {"bucket": 16, "block_m": 16, "block_n": 64, "num_warps": 8, "num_stages": 1, "split_k": 8, "reduce_block_n": 64, "reduce_warps": 4},
    "v7_m16_n64_w8_s1_sp8_r128w8": {"bucket": 16, "block_m": 16, "block_n": 64, "num_warps": 8, "num_stages": 1, "split_k": 8, "reduce_block_n": 128, "reduce_warps": 8},
    "v7_m16_n64_w4_s2_sp8_r32w2": {"bucket": 16, "block_m": 16, "block_n": 64, "num_warps": 4, "num_stages": 2, "split_k": 8, "reduce_block_n": 32, "reduce_warps": 2},
}
_ALLOWED_CONFIG_KEYS = {"bucket", "block_m", "block_n", "num_warps", "num_stages", "split_k", "reduce_block_n", "reduce_warps"}


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
    cid = next(name for name, config in CONFIGS.items() if config["bucket"] == bucket and config["split_k"] == 1)
    return dict(CONFIGS[cid])


def _resolve_config(config: str | dict[str, Any] | None, bucket: int, *, allow_diagnostic: bool = False) -> dict[str, int]:
    if isinstance(config, str):
        if config not in CONFIGS:
            raise ValueError(f"unknown config id {config!r}")
        result = dict(CONFIGS[config])
    else:
        result = common.normalize_config(config, _default_config(bucket), _ALLOWED_CONFIG_KEYS)
    if result["bucket"] != bucket:
        raise ValueError("config bucket mismatch")
    if result["block_n"] not in (32, 64) or result["num_warps"] not in (2, 4, 8):
        raise ValueError("unsupported bounded producer schedule")
    if result["reduce_block_n"] not in (32, 64, 128) or result["reduce_warps"] not in (2, 4, 8):
        raise ValueError("unsupported bounded reducer schedule")
    if result["num_stages"] not in (1, 2, 3):
        raise ValueError("num_stages must be 1, 2, or 3")
    if result["block_m"] not in (8, 16) or result["block_m"] < bucket:
        raise ValueError("block_m must cover the logical bucket")
    base = {key: result[key] for key in ("bucket", "block_m", "block_n", "num_warps", "num_stages", "reduce_block_n", "reduce_warps")}
    if not any(all(candidate[key] == value for key, value in base.items()) for candidate in CONFIGS.values()):
        raise ValueError(f"schedule is outside the bounded manifest: {base}")
    if not allow_diagnostic and result not in CONFIGS.values():
        raise ValueError("public run accepts exactly the 12 declared config dictionaries")
    return result


def _run_impl(x, qdata, scale, scale_step, zero, zero_step, *, bucket: int, config=None, allow_diagnostic: bool = False) -> torch.Tensor:
    problem = common.validate_inputs(x, qdata, scale, scale_step, zero, zero_step, bucket=bucket)
    chosen = _resolve_config(config, bucket, allow_diagnostic=allow_diagnostic)
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
        BLOCK_M=chosen["block_m"], BLOCK_N=chosen["block_n"], PIPELINE_STAGES=chosen["num_stages"],
        num_warps=chosen["num_warps"], num_stages=chosen["num_stages"],
    )
    if split_k > 1:
        launch.update(stride_os=split_stride, SPLIT_K=split_k)
    kernel[grid](x, qdata, scale, scale_step, zero, zero_step, destination, problem.m, **launch)
    common.finish_split(destination, out, split_k, block_n=chosen["reduce_block_n"], num_warps=chosen["reduce_warps"])
    return out


def run(x, qdata, scale, scale_step, zero, zero_step, *, bucket: int, config=None) -> torch.Tensor:
    return _run_impl(x, qdata, scale, scale_step, zero, zero_step, bucket=bucket, config=config)


def _run_diagnostic(x, qdata, scale, scale_step, zero, zero_step, *, bucket: int, config) -> torch.Tensor:
    return _run_impl(x, qdata, scale, scale_step, zero, zero_step,
                     bucket=bucket, config=config, allow_diagnostic=True)


_DIRECT_KERNEL = _precision_direct_kernel
_SPLIT_KERNEL = _precision_splitk_kernel
__all__ = ["CONFIGS", "run"]
