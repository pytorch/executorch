"""Opt-in dynamic AOTI trial for Gemma FP32-decoded A16 prefill.

Gate M128/M512 and Down M128 apply only to this trial.
Down M512 remains original after its compiled performance regression. All other
sizes use the original ET operation. Every candidate invocation prepares
temporary group metadata from the original packing. No model registration.
"""
import torch
from torch.library import triton_op, wrap_triton
import triton
from executorch.backends.cuda.quantize_op_dispatch._gemm_family_dispatch import chunked_dequant_linear
from public_grouped_w4a16 import _validate
from public_prefill_gemma_fp32 import _prepare, _gemm


def reference_fp32(x,qdata,scale,scale_step,zero,zero_step,group_size):
    """Exact selected Gemma weight decode arithmetic; original input bytes."""
    n,khalf=qdata.shape;k=khalf*2;groups=k//group_size;gps=256//group_size
    def rows(i,j):
        packed=qdata[i:j].reshape(j-i,groups,group_size//2)
        q=torch.stack(((packed&15).float(),(packed>>4).float()),dim=-1).reshape(j-i,groups,group_size)
        s=scale[i:j].float()*scale_step[i:j].float().repeat_interleave(gps,dim=1)
        z=zero[i:j].float()*zero_step[i:j].float().repeat_interleave(gps,dim=1)
        dense=((q-z[:,:,None])*s[:,:,None]).reshape(j-i,k).to(x.dtype)
        return torch.nn.functional.linear(x,dense)
    return chunked_dequant_linear(x,n,rows)


@triton_op('siliconbench::public_gemma_fp32_group_gemm', mutates_args={})
def prepared_candidate_op(x: torch.Tensor, q: torch.Tensor, sc: torch.Tensor,
                          ss: torch.Tensor, z: torch.Tensor, zs: torch.Tensor) -> torch.Tensor:
    m, n, k = _validate(x, (q, sc, ss, z, zs), 'bulk-staged-bf16')
    if (n, k) == (30720, 3840):
        bn, stages = 128, 2
    else:
        assert (n, k) == (3840, 15360)
        bn, stages = 64, 3
    sg = torch.empty((n, k // 32), device=x.device, dtype=torch.float32)
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


@triton_op('siliconbench::public_gemma_fp32_prefill_a16', mutates_args={})
def prepared_prefill_op(x: torch.Tensor, q: torch.Tensor, sc: torch.Tensor,
                        ss: torch.Tensor, z: torch.Tensor, zs: torch.Tensor) -> torch.Tensor:
    # Compiled AOTI measured Down512 slower than the original operation.
    # Preserve that original path; retain the verified Down128 candidate.
    if (q.shape[0],q.shape[1]*2)==(3840,15360):
        return torch.cond(x.shape[0]==128,
            lambda a,b,c,d,e,f:prepared_candidate_op(a,b,c,d,e,f),
            lambda a,b,c,d,e,f:reference_fp32(a,b,c,d,e,f,32),
            (x,q,sc,ss,z,zs))
    # Inductor's current symbolic lowering rejects the compound SymPy Or.
    # Use the same nested exact-size conditions as the verified dense trial.
    return torch.cond(x.shape[0] == 128,
        lambda a,b,c,d,e,f: prepared_candidate_op(a,b,c,d,e,f),
        lambda a,b,c,d,e,f: torch.cond(a.shape[0] == 512,
            lambda g,h,i,j,l,p: prepared_candidate_op(g,h,i,j,l,p),
            lambda g,h,i,j,l,p: reference_fp32(g,h,i,j,l,p,32),
            (a,b,c,d,e,f)),
        (x,q,sc,ss,z,zs))


@prepared_prefill_op.register_fake
def _prefill_fake(x, q, sc, ss, z, zs):
    return torch.empty((x.shape[0], q.shape[0]), device=x.device, dtype=torch.bfloat16)
