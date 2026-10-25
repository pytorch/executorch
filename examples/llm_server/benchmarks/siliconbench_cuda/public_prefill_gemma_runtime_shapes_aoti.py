"""Opt-in Gate prefill routing for actual 16-session server batch sizes.

The existing worker reserves 16 tokens from a 512-token scheduler budget, so
its full prefill chunk is 496 tokens. One full chunk plus active decodes can
produce M496..511. Keep the same FP32 weight reconstruction, BF16 operands,
prepared metadata kernel, packed weights and static methods. Every other M
retains the reference operation. This remains an unqualified experiment.
"""
import torch
from torch.library import triton_op
from public_prefill_gemma_fp32_aoti import prepared_candidate_op, reference_fp32


@triton_op('siliconbench::public_gemma_runtime_prefill_a16',mutates_args={})
def prepared_runtime_prefill_op(x:torch.Tensor,q:torch.Tensor,sc:torch.Tensor,
                               ss:torch.Tensor,z:torch.Tensor,zs:torch.Tensor)->torch.Tensor:
    shape=(q.shape[0],q.shape[1]*2)
    if shape==(3840,15360):
        return torch.cond(x.shape[0]==128,
            lambda a,b,c,d,e,f:prepared_candidate_op(a,b,c,d,e,f),
            lambda a,b,c,d,e,f:reference_fp32(a,b,c,d,e,f,32),
            (x,q,sc,ss,z,zs))
    assert shape==(30720,3840)
    # Single comparisons avoid the symbolic compound-predicate lowering that
    # failed in the earlier isolated AOTI trial. Bound both ends explicitly.
    return torch.cond(x.shape[0]==128,
        lambda a,b,c,d,e,f:prepared_candidate_op(a,b,c,d,e,f),
        lambda a,b,c,d,e,f:torch.cond(a.shape[0]>=496,
            lambda g,h,i,j,k,l:torch.cond(g.shape[0]<=512,
                lambda u,v,w,y,t,r:prepared_candidate_op(u,v,w,y,t,r),
                lambda u,v,w,y,t,r:reference_fp32(u,v,w,y,t,r,32),
                (g,h,i,j,k,l)),
            lambda g,h,i,j,k,l:reference_fp32(g,h,i,j,k,l,32),
            (a,b,c,d,e,f)),
        (x,q,sc,ss,z,zs))


@prepared_runtime_prefill_op.register_fake
def _fake(x,q,sc,ss,z,zs):
    return torch.empty((x.shape[0],q.shape[0]),device=x.device,dtype=torch.bfloat16)
