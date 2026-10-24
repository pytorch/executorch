# Optimized by KernelAgent-Oink(https://github.com/meta-pytorch/KernelAgent)
"""Strict packed-W4 contract and configurable deterministic split reduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import triton
import triton.language as tl

GROUP_SIZE = 32
SUPER_BLOCK = 256
SUPPORTED_SPLITS = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class Problem:
    m: int
    n: int
    k: int
    bucket: int
    device: torch.device


def validate_inputs(x, qdata, scale, scale_step, zero, zero_step, *, bucket: int) -> Problem:
    if bucket not in (8, 16):
        raise ValueError(f"v7 bucket must be 8 or 16, got {bucket}")
    if any(t.ndim != 2 for t in (x, qdata, scale, scale_step, zero, zero_step)):
        raise ValueError("all inputs must be rank-2")
    m, k = x.shape
    n, k_half = qdata.shape
    if not 1 <= m <= bucket or n <= 0 or k <= 0 or k % SUPER_BLOCK:
        raise ValueError(f"invalid M/N/K for bucket {bucket}: {(m, n, k)}")
    if k_half * 2 != k:
        raise ValueError("qdata must have shape [N,K/2]")
    if tuple(scale.shape) != (n, k // GROUP_SIZE) or tuple(zero.shape) != (n, k // GROUP_SIZE):
        raise ValueError("scale/zero must have shape [N,K/32]")
    if tuple(scale_step.shape) != (n, k // SUPER_BLOCK) or tuple(zero_step.shape) != (n, k // SUPER_BLOCK):
        raise ValueError("scale_step/zero_step must have shape [N,K/256]")
    expected = (("x", x, torch.bfloat16), ("qdata", qdata, torch.uint8),
                ("scale", scale, torch.uint8), ("scale_step", scale_step, torch.float16),
                ("zero", zero, torch.uint8), ("zero_step", zero_step, torch.float16))
    for name, tensor, dtype in expected:
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must be {dtype}, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.device != x.device:
            raise ValueError("all inputs must share one device")
    if x.device.type != "cuda":
        raise ValueError("CUDA tensors are required")
    return Problem(m, n, k, bucket, x.device)


def validate_split(split_k: int, k: int) -> None:
    if split_k not in SUPPORTED_SPLITS:
        raise ValueError(f"split_k must be one of {SUPPORTED_SPLITS}")
    if split_k > k // SUPER_BLOCK:
        raise ValueError(f"split_k={split_k} exceeds {k // SUPER_BLOCK} K256 blocks")


def balanced_super_ranges(k: int, split_k: int) -> list[tuple[int, int]]:
    validate_split(split_k, k)
    blocks = k // SUPER_BLOCK
    base, extra = divmod(blocks, split_k)
    first, result = 0, []
    for split_id in range(split_k):
        count = base + (split_id < extra)
        result.append((first, first + count))
        first += count
    return result


def normalize_config(config: dict[str, Any] | None, defaults: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    result = dict(defaults)
    if config is not None:
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        result.update(config)
    return result


@triton.jit
def _splitk_reduce_kernel(partial, out, m, N: tl.constexpr,
                          stride_ps: tl.constexpr, stride_pm: tl.constexpr, stride_pn: tl.constexpr,
                          stride_om: tl.constexpr, stride_on: tl.constexpr,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = (offs_m[:, None] < m) & (offs_n[None, :] < N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for split_id in range(SPLIT_K):
        acc += tl.load(partial + split_id * stride_ps + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn,
                       mask=mask, other=0.0)
    tl.store(out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc.to(tl.bfloat16), mask=mask)


def allocate_output(problem: Problem, split_k: int):
    out = torch.empty((problem.m, problem.n), dtype=torch.bfloat16, device=problem.device)
    if split_k == 1:
        return out, out, 0
    partial = torch.empty((split_k, problem.bucket, problem.n), dtype=torch.float32, device=problem.device)
    return out, partial, partial.stride(0)


def finish_split(partial, out, split_k: int, *, block_n: int, num_warps: int) -> None:
    if split_k == 1:
        return
    m, n = out.shape
    block_m = triton.next_power_of_2(m)
    _splitk_reduce_kernel[(triton.cdiv(n, block_n), triton.cdiv(m, block_m))](
        partial, out, m, N=n, stride_ps=partial.stride(0), stride_pm=partial.stride(1),
        stride_pn=partial.stride(2), stride_om=out.stride(0), stride_on=out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, SPLIT_K=split_k,
        num_warps=num_warps, num_stages=1)


def weight_strides(qdata, scale, scale_step, zero, zero_step) -> tuple[int, ...]:
    return (qdata.stride(0), qdata.stride(1), scale.stride(0), scale.stride(1),
            scale_step.stride(0), scale_step.stride(1), zero.stride(0), zero.stride(1),
            zero_step.stride(0), zero_step.stride(1))


_REDUCE_KERNEL = _splitk_reduce_kernel
