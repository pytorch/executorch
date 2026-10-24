# Optimized by KernelAgent-Oink(https://github.com/meta-pytorch/KernelAgent)
"""Traceable Triton custom-op boundary for explicit pending v9 configurations."""
from __future__ import annotations
import torch
import triton
from torch.library import triton_op, wrap_triton
from .kernels import CONFIGS, _DIRECT_KERNEL, _SPLIT_KERNEL
from .common import _REDUCE_KERNEL

CONFIG_IDS = (
    "v9_m8_attn_control_sp8_r32",
    "v9_m8_down_native_sp8_r32",
    "v9_m8_down_native_sp4_r32",
    "v9_m16_down_native_sp8_r64",
    "v9_m16_down_native_sp4_r64",
    "v9_m16_fused_native_sp1",
    "v9_m8_fused_native_sp1",
    "v9_m8_mg_atomic_native_sp2",
    "v9_m8_mg_fused_native_sp2",
    "v9_m16_mg_native_sp1",
)
if tuple(CONFIGS) != CONFIG_IDS:
    raise RuntimeError("v9 config-index ABI differs from the manifest order")

@triton_op("kernelagent::w4a16_v9", mutates_args={})
def candidate_op(x: torch.Tensor, qdata: torch.Tensor, scale: torch.Tensor,
                 scale_step: torch.Tensor, zero: torch.Tensor, zero_step: torch.Tensor,
                 config_index: int) -> torch.Tensor:
    if not 0 <= config_index < len(CONFIG_IDS):
        raise ValueError("invalid v9 config index")
    tensors = (x, qdata, scale, scale_step, zero, zero_step)
    if any(t.ndim != 2 for t in tensors):
        raise ValueError("all v9 traceable inputs must be rank-2")
    config_id = CONFIG_IDS[config_index]
    c = CONFIGS[config_id]
    m, k = x.shape
    n = qdata.shape[0]
    min_m = 5 if c["bucket"] == 8 else 9
    if (not min_m <= m <= c["bucket"] or n <= 0 or k <= 0 or k % 256
            or qdata.shape != (n, k // 2)):
        raise ValueError("invalid v9 traceable tensor shapes")
    if scale.shape != (n, k // 32) or zero.shape != (n, k // 32):
        raise ValueError("invalid scale/zero shape")
    if scale_step.shape != (n, k // 256) or zero_step.shape != (n, k // 256):
        raise ValueError("invalid step shape")
    expected = ((x, torch.bfloat16), (qdata, torch.uint8), (scale, torch.uint8),
                (scale_step, torch.float16), (zero, torch.uint8), (zero_step, torch.float16))
    if x.device.type != "cuda" or any(
            t.dtype != dtype or t.device != x.device or not t.is_contiguous()
            for t, dtype in expected):
        raise ValueError("invalid v9 traceable tensor contract")
    if c["split_k"] > k // 256:
        raise ValueError("split exceeds K256 block count")
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    split = c["split_k"]
    destination = out if split == 1 else torch.empty((split, c["bucket"], n), dtype=torch.float32, device=x.device)
    split_stride = 0 if split == 1 else destination.stride(0)
    kernel = _DIRECT_KERNEL if split == 1 else _SPLIT_KERNEL
    grid = (triton.cdiv(n, c["block_n"]),) if split == 1 else (triton.cdiv(n, c["block_n"]), split)
    launch = dict(N=n, K=k, stride_xm=x.stride(0), stride_xk=x.stride(1),
                  stride_qn=qdata.stride(0), stride_qk=qdata.stride(1),
                  stride_sn=scale.stride(0), stride_sk=scale.stride(1),
                  stride_ssn=scale_step.stride(0), stride_ssk=scale_step.stride(1),
                  stride_zn=zero.stride(0), stride_zk=zero.stride(1),
                  stride_zsn=zero_step.stride(0), stride_zsk=zero_step.stride(1),
                  stride_om=destination.stride(-2), stride_on=destination.stride(-1),
                  BLOCK_M=c["block_m"], BLOCK_N=c["block_n"], PIPELINE_STAGES=c["num_stages"],
                  num_warps=c["num_warps"], num_stages=c["num_stages"])
    if split > 1:
        launch.update(stride_os=split_stride, SPLIT_K=split)
    wrap_triton(kernel)[grid](x, qdata, scale, scale_step, zero, zero_step, destination, m, **launch)
    if split > 1:
        rb, rw = c["reduce_block_n"], c["reduce_warps"]
        wrap_triton(_REDUCE_KERNEL)[(triton.cdiv(n, rb), 1)](
            destination, out, m, N=n, stride_ps=destination.stride(0),
            stride_pm=destination.stride(1), stride_pn=destination.stride(2),
            stride_om=out.stride(0), stride_on=out.stride(1),
            BLOCK_M=c["block_m"], BLOCK_N=rb, SPLIT_K=split,
            num_warps=rw, num_stages=1)
    return out

@candidate_op.register_fake
def _candidate_fake(x, qdata, scale, scale_step, zero, zero_step, config_index: int):
    if not 0 <= config_index < len(CONFIG_IDS):
        raise ValueError("invalid v9 config index")
    tensors = (x, qdata, scale, scale_step, zero, zero_step)
    if any(t.ndim != 2 for t in tensors):
        raise ValueError("all v9 fake inputs must be rank-2")
    c = CONFIGS[CONFIG_IDS[config_index]]
    m, k = x.shape
    n = qdata.shape[0]
    min_m = 5 if c["bucket"] == 8 else 9
    expected_shapes = ((m, k), (n, k // 2), (n, k // 32), (n, k // 256),
                       (n, k // 32), (n, k // 256))
    expected_dtypes = (torch.bfloat16, torch.uint8, torch.uint8, torch.float16,
                       torch.uint8, torch.float16)
    if (not min_m <= m <= c["bucket"] or n <= 0 or k <= 0 or k % 256
            or c["split_k"] > k // 256 or x.device.type != "cuda"):
        raise ValueError("invalid v9 fake tensor contract")
    if any(t.shape != shape or t.dtype != dtype or t.device != x.device or not t.is_contiguous()
           for t, shape, dtype in zip(tensors, expected_shapes, expected_dtypes)):
        raise ValueError("invalid v9 fake tensor metadata")
    return torch.empty((m, n), dtype=torch.bfloat16, device=x.device)

def config_index(config_id: str) -> int:
    try:
        return CONFIG_IDS.index(config_id)
    except ValueError as error:
        raise ValueError(f"unknown v9 config {config_id!r}") from error

__all__ = ["CONFIG_IDS", "candidate_op", "config_index"]
