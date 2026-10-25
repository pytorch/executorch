"""Explicit AOTI boundary for the standalone public W4A16 experiment.

Importing this module registers only a new experimental operator. It does
not register an ET quantized tensor implementation or alter model dispatch.
Numerical/performance gates and an SM120-only model policy are separate.
"""
from dataclasses import asdict

import torch
from torch.library import triton_op, wrap_triton
import triton

from public_grouped_w4a16 import _gemm, _reduce, _validate, configs

CONFIGS = tuple(configs(72))


@triton_op('siliconbench::public_grouped_w4a16', mutates_args={})
def candidate_op(x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor,
                 scale_step: torch.Tensor, zero: torch.Tensor,
                 zero_step: torch.Tensor, config_index: int,
                 native_zero_fma: bool) -> torch.Tensor:
    if not 0 <= config_index < len(CONFIGS):
        raise ValueError('Unknown experimental configuration')
    cfg = CONFIGS[config_index].validate()
    arithmetic = 'native-zero-fma-bf16' if native_zero_fma else 'bulk-staged-bf16'
    m, n, k = _validate(x, (q, scale, scale_step, zero, zero_step), arithmetic)
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    partial = (torch.empty((cfg.split, m, n), device=x.device, dtype=torch.float32)
               if cfg.split > 1 else out)
    grid = (triton.cdiv(m, cfg.bm) * triton.cdiv(n, cfg.bn), cfg.split)
    wrap_triton(_gemm)[grid](x, q, scale, scale_step, zero, zero_step, partial, m,
        N=n, K=k, BM=cfg.bm, BN=cfg.bn, BK=cfg.bk, SPLIT=cfg.split,
        GROUP_M=cfg.group_m, STAGES=cfg.stages, ZERO_FMA=native_zero_fma,
        num_warps=cfg.warps, num_stages=cfg.stages, enable_fp_fusion=False)
    if cfg.split > 1:
        wrap_triton(_reduce)[(triton.cdiv(m*n, 256),)](partial, out, m*n,
            SPLIT=cfg.split, BLOCK=256, num_warps=4, enable_fp_fusion=False)
    return out


@candidate_op.register_fake
def _fake(x, q, scale, scale_step, zero, zero_step, config_index: int, native_zero_fma: bool):
    if not 0 <= config_index < len(CONFIGS):
        raise ValueError('Unknown experimental configuration')
    arithmetic = 'native-zero-fma-bf16' if native_zero_fma else 'bulk-staged-bf16'
    m, n, k = _validate(x, (q, scale, scale_step, zero, zero_step), arithmetic)
    return torch.empty((m, n), device=x.device, dtype=torch.bfloat16)


def config_index(config):
    data = asdict(config) if hasattr(config, '__dataclass_fields__') else dict(config)
    return next(i for i, cfg in enumerate(CONFIGS) if asdict(cfg) == data)
