"""Opt-in dynamic AOTI trial for exact prepared-metadata A16 prefill.

Measured M512 configurations apply only to the M512/M520 trial. All other
sizes use the original ET operation. Every candidate invocation prepares
temporary group metadata from the original packing. No model registration.
"""
import torch
from torch.library import triton_op, wrap_triton
import triton
from executorch.backends.cuda.quantize_op_dispatch.int4_dispatch import _dequant_matmul
from public_grouped_w4a16 import _validate
from public_prefill_prepared_groups import _prepare, _gemm


@triton_op('siliconbench::public_prepared_group_gemm', mutates_args={})
def prepared_candidate_op(x: torch.Tensor, q: torch.Tensor, sc: torch.Tensor,
                          ss: torch.Tensor, z: torch.Tensor, zs: torch.Tensor) -> torch.Tensor:
    m, n, k = _validate(x, (q, sc, ss, z, zs), 'bulk-staged-bf16')
    if (n, k) == (39936, 6656):
        bn, stages = 128, 2
    else:
        assert (n, k) == (6656, 19968)
        bn, stages = 64, 3
    sg = torch.empty((n, k // 32), device=x.device, dtype=torch.bfloat16)
    zg = torch.empty_like(sg)
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    wrap_triton(_prepare)[(triton.cdiv(n * (k // 32), 256),)](
        sc, ss, z, zs, sg, zg, SIZE=n * (k // 32), GROUPS=k // 32,
        BLOCK=256, num_warps=4, num_stages=1, enable_fp_fusion=False)
    wrap_triton(_gemm)[(triton.cdiv(m, 64) * triton.cdiv(n, bn),)](
        x, q, sg, zg, out, m, N=n, K=k, BM=64, BN=bn, BK=128,
        GROUP_M=8, STAGES=stages, num_warps=4, num_stages=stages,
        enable_fp_fusion=False)
    return out


@prepared_candidate_op.register_fake
def _candidate_fake(x, q, sc, ss, z, zs):
    return torch.empty((x.shape[0], q.shape[0]), device=x.device, dtype=torch.bfloat16)


@triton_op('siliconbench::public_prepared_prefill_a16', mutates_args={})
def prepared_prefill_op(x: torch.Tensor, q: torch.Tensor, sc: torch.Tensor,
                        ss: torch.Tensor, z: torch.Tensor, zs: torch.Tensor) -> torch.Tensor:
    return torch.cond((x.shape[0] == 512) | (x.shape[0] == 520),
        lambda a,b,c,d,e,f: prepared_candidate_op(a,b,c,d,e,f),
        lambda a,b,c,d,e,f: _dequant_matmul(a,b,c,d,e,f,32),
        (x,q,sc,ss,z,zs))


@prepared_prefill_op.register_fake
def _prefill_fake(x, q, sc, ss, z, zs):
    return torch.empty((x.shape[0], q.shape[0]), device=x.device, dtype=torch.bfloat16)
