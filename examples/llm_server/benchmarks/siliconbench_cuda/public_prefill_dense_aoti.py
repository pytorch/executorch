"""Opt-in AOTI boundary for two measured MG Down prefill configurations.

Read the original ET Q4 packing with staged BF16 reconstruction. The regular
grouped-grid BF16 GEMM follows the existing public_prefill_dense adaptation
of Triton's tutorial. Only M512 and M520 use measured configurations; other
sizes keep the original ET bulk operation. This does not change dispatch.
"""
import torch
from torch.library import triton_op, wrap_triton
import triton
from executorch.backends.cuda.quantize_op_dispatch.int4_dispatch import _dequant_matmul
from public_grouped_w4a16 import _materialize, _validate
from public_prefill_dense import _dense


def _pipeline(x, q, sc, ss, z, zs, bn):
    m, n, k = _validate(x, (q, sc, ss, z, zs), 'bulk-staged-bf16')
    assert (n, k) == (6656, 19968)
    weight = torch.empty((n, k), device=x.device, dtype=torch.bfloat16)
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    wrap_triton(_materialize)[(triton.cdiv(n, 32), k//128)](
        q, sc, ss, z, zs, weight, N=n, K=k, BN=32, BK=128,
        ZERO_FMA=False, num_warps=4, num_stages=1, enable_fp_fusion=False)
    tiles = triton.cdiv(m, 64) * triton.cdiv(n, bn)
    wrap_triton(_dense)[(tiles,)](x, weight, out, m, N=n, K=k,
        BM=64, BN=bn, BK=128, GROUP=8, STRIDE=tiles, STAGES=3,
        num_warps=4, num_stages=3, enable_fp_fusion=False)
    return out


@triton_op('siliconbench::public_down_prefill_bf16', mutates_args={})
def down_prefill_op(x: torch.Tensor, q: torch.Tensor, sc: torch.Tensor,
                    ss: torch.Tensor, z: torch.Tensor, zs: torch.Tensor) -> torch.Tensor:
    return torch.cond(x.shape[0] == 512,
        lambda a,b,c,d,e,f: _pipeline(a,b,c,d,e,f,64),
        lambda a,b,c,d,e,f: torch.cond(a.shape[0] == 520,
            lambda g,h,i,j,l,p: _pipeline(g,h,i,j,l,p,128),
            lambda g,h,i,j,l,p: _dequant_matmul(g,h,i,j,l,p,32),
            (a,b,c,d,e,f)),
        (x,q,sc,ss,z,zs))


@down_prefill_op.register_fake
def _fake(x, q, sc, ss, z, zs):
    m, n, k = _validate(x, (q, sc, ss, z, zs), 'bulk-staged-bf16')
    assert (n, k) == (6656, 19968)
    return torch.empty((m,n), dtype=torch.bfloat16, device=x.device)
