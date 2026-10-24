# Optimized by KernelAgent-Oink(https://github.com/meta-pytorch/KernelAgent)
"""Planar Q6_K W6A16 K256/K128 schedules with native staged-BF16 rounding."""

from __future__ import annotations
from typing import Any
import torch
import triton
import triton.language as tl
from . import common

CONFIGS:dict[str,dict[str,int]]={
 "w6v6_m8_k256_n32_w2_s1_sp1":{"bucket":8,"fragment_k":256,"block_n":32,"num_warps":2,"num_stages":1,"split_k":1},
 "w6v6_m8_k256_n64_w2_s1_sp2":{"bucket":8,"fragment_k":256,"block_n":64,"num_warps":2,"num_stages":1,"split_k":2},
 "w6v6_m8_k128_n64_w4_s2_sp1":{"bucket":8,"fragment_k":128,"block_n":64,"num_warps":4,"num_stages":2,"split_k":1},
 "w6v6_m8_k256_n32_w2_s1_sp8":{"bucket":8,"fragment_k":256,"block_n":32,"num_warps":2,"num_stages":1,"split_k":8},
 "w6v6_m8_k256_n64_w2_s1_sp8":{"bucket":8,"fragment_k":256,"block_n":64,"num_warps":2,"num_stages":1,"split_k":8},
 "w6v6_m8_k256_n32_w4_s1_sp8":{"bucket":8,"fragment_k":256,"block_n":32,"num_warps":4,"num_stages":1,"split_k":8},
 "w6v6_m8_k128_n32_w4_s2_sp8":{"bucket":8,"fragment_k":128,"block_n":32,"num_warps":4,"num_stages":2,"split_k":8},
 "w6v6_m8_k256_n32_w4_s1_sp16":{"bucket":8,"fragment_k":256,"block_n":32,"num_warps":4,"num_stages":1,"split_k":16},
 "w6v6_m16_k256_n32_w2_s1_sp1":{"bucket":16,"fragment_k":256,"block_n":32,"num_warps":2,"num_stages":1,"split_k":1},
 "w6v6_m16_k256_n64_w2_s1_sp2":{"bucket":16,"fragment_k":256,"block_n":64,"num_warps":2,"num_stages":1,"split_k":2},
 "w6v6_m16_k256_n64_w4_s1_sp1":{"bucket":16,"fragment_k":256,"block_n":64,"num_warps":4,"num_stages":1,"split_k":1},
 "w6v6_m16_k128_n64_w4_s2_sp1":{"bucket":16,"fragment_k":128,"block_n":64,"num_warps":4,"num_stages":2,"split_k":1},
 "w6v6_m16_k256_n32_w2_s1_sp8":{"bucket":16,"fragment_k":256,"block_n":32,"num_warps":2,"num_stages":1,"split_k":8},
 "w6v6_m16_k256_n64_w2_s1_sp8":{"bucket":16,"fragment_k":256,"block_n":64,"num_warps":2,"num_stages":1,"split_k":8},
 "w6v6_m16_k256_n64_w8_s1_sp8":{"bucket":16,"fragment_k":256,"block_n":64,"num_warps":8,"num_stages":1,"split_k":8},
 "w6v6_m16_k128_n64_w4_s2_sp8":{"bucket":16,"fragment_k":128,"block_n":64,"num_warps":4,"num_stages":2,"split_k":8},
}
_ALLOWED={"bucket","fragment_k","block_n","num_warps","num_stages","split_k"}


@triton.jit
def _signed_byte(value):
    return value.to(tl.int8,bitcast=True).to(tl.float32)

@triton.jit
def _decode_fragment(ql,qh,scale,step_value,offs_n,n_mask,super_block,fragment_id:tl.constexpr,stride_qln:tl.constexpr,stride_qlk:tl.constexpr,stride_qhn:tl.constexpr,stride_qhk:tl.constexpr,stride_sn:tl.constexpr,stride_sk:tl.constexpr,BLOCK_N:tl.constexpr,FRAGMENT_K:tl.constexpr):
    local_k=fragment_id*FRAGMENT_K+tl.arange(0,FRAGMENT_K)
    packed_low=tl.load(ql+offs_n[:,None]*stride_qln+(super_block*128+local_k[None,:]//2)*stride_qlk,mask=n_mask[:,None],other=0).to(tl.uint32)
    low=(packed_low>>((local_k[None,:]&1)*4))&0xF
    k32=local_k%32
    qh_byte=super_block*64+(local_k//32)*8+(k32%2)*4+k32//8
    high_shift=((k32%8)//2)*2
    packed_high=tl.load(qh+offs_n[:,None]*stride_qhn+qh_byte[None,:]*stride_qhk,mask=n_mask[:,None],other=0).to(tl.uint32)
    high=(packed_high>>high_shift[None,:])&0x3
    q=(low|(high<<4)).to(tl.int32)-32
    groups:tl.constexpr=FRAGMENT_K//16
    group=super_block*16+fragment_id*groups+tl.arange(0,groups)
    code=tl.load(scale+offs_n[:,None]*stride_sn+group[None,:]*stride_sk,mask=n_mask[:,None],other=0)
    code=_signed_byte(code).to(tl.bfloat16)
    gs=(code*step_value[:,None]).to(tl.bfloat16)
    expanded=tl.reshape(tl.broadcast_to(gs[:,:,None],(BLOCK_N,groups,16)),(BLOCK_N,FRAGMENT_K),can_reorder=False)
    return (q.to(tl.bfloat16)*expanded).to(tl.bfloat16)

@triton.jit
def _direct(x,ql,qh,scale,steps,out,M,N:tl.constexpr,K:tl.constexpr,stride_xm:tl.constexpr,stride_xk:tl.constexpr,stride_qln:tl.constexpr,stride_qlk:tl.constexpr,stride_qhn:tl.constexpr,stride_qhk:tl.constexpr,stride_sn:tl.constexpr,stride_sk:tl.constexpr,stride_stn:tl.constexpr,stride_stk:tl.constexpr,stride_om:tl.constexpr,stride_on:tl.constexpr,BLOCK_M:tl.constexpr,BLOCK_N:tl.constexpr,FRAGMENT_K:tl.constexpr,PIPELINE_STAGES:tl.constexpr):
    pn=tl.program_id(0); om=tl.arange(0,BLOCK_M); on=pn*BLOCK_N+tl.arange(0,BLOCK_N); mm=om<M; nm=on<N; ok=tl.arange(0,FRAGMENT_K); acc=tl.zeros((BLOCK_M,BLOCK_N),tl.float32); fragments:tl.constexpr=256//FRAGMENT_K
    for sb in tl.range(0,K//256,num_stages=PIPELINE_STAGES):
        step=tl.load(steps+on*stride_stn+sb*stride_stk,mask=nm,other=0.0).to(tl.bfloat16)
        for frag in tl.static_range(0,fragments):
            w=_decode_fragment(ql,qh,scale,step,on,nm,sb,frag,stride_qln,stride_qlk,stride_qhn,stride_qhk,stride_sn,stride_sk,BLOCK_N,FRAGMENT_K)
            a=tl.load(x+om[:,None]*stride_xm+(sb*256+frag*FRAGMENT_K+ok[None,:])*stride_xk,mask=mm[:,None],other=0.0).to(tl.bfloat16)
            acc+=tl.dot(a,tl.trans(w),out_dtype=tl.float32)
    tl.store(out+om[:,None]*stride_om+on[None,:]*stride_on,acc.to(tl.bfloat16),mask=mm[:,None]&nm[None,:])

@triton.jit
def _split(x,ql,qh,scale,steps,out,M,N:tl.constexpr,K:tl.constexpr,stride_xm:tl.constexpr,stride_xk:tl.constexpr,stride_qln:tl.constexpr,stride_qlk:tl.constexpr,stride_qhn:tl.constexpr,stride_qhk:tl.constexpr,stride_sn:tl.constexpr,stride_sk:tl.constexpr,stride_stn:tl.constexpr,stride_stk:tl.constexpr,stride_os:tl.constexpr,stride_om:tl.constexpr,stride_on:tl.constexpr,BLOCK_M:tl.constexpr,BLOCK_N:tl.constexpr,FRAGMENT_K:tl.constexpr,SPLIT_K:tl.constexpr,PIPELINE_STAGES:tl.constexpr):
    pn=tl.program_id(0); sid=tl.program_id(1); om=tl.arange(0,BLOCK_M); on=pn*BLOCK_N+tl.arange(0,BLOCK_N); mm=om<M; nm=on<N; ok=tl.arange(0,FRAGMENT_K); acc=tl.zeros((BLOCK_M,BLOCK_N),tl.float32); fragments:tl.constexpr=256//FRAGMENT_K; count:tl.constexpr=tl.cdiv(K//256,SPLIT_K); first=sid*count; last=tl.minimum(first+count,K//256)
    for sb in tl.range(first,last,num_stages=PIPELINE_STAGES):
        step=tl.load(steps+on*stride_stn+sb*stride_stk,mask=nm,other=0.0).to(tl.bfloat16)
        for frag in tl.static_range(0,fragments):
            w=_decode_fragment(ql,qh,scale,step,on,nm,sb,frag,stride_qln,stride_qlk,stride_qhn,stride_qhk,stride_sn,stride_sk,BLOCK_N,FRAGMENT_K)
            a=tl.load(x+om[:,None]*stride_xm+(sb*256+frag*FRAGMENT_K+ok[None,:])*stride_xk,mask=mm[:,None],other=0.0).to(tl.bfloat16)
            acc+=tl.dot(a,tl.trans(w),out_dtype=tl.float32)
    tl.store(out+sid*stride_os+om[:,None]*stride_om+on[None,:]*stride_on,acc,mask=mm[:,None]&nm[None,:])


def _resolve(config,bucket:int,allow_diagnostic:bool=False):
    if isinstance(config,str):
        if config not in CONFIGS: raise ValueError(f"unknown config {config}")
        out=dict(CONFIGS[config])
    else:
        default=next(dict(c) for c in CONFIGS.values() if c["bucket"]==bucket and c["split_k"]==1)
        out=common.normalize_config(config,default,_ALLOWED)
    if out["bucket"]!=bucket or out["fragment_k"] not in (128,256) or out["block_n"] not in (32,64) or out["num_warps"] not in (2,4,8) or out["num_stages"] not in (1,2): raise ValueError("unsupported schedule")
    base={k:out[k] for k in ("bucket","fragment_k","block_n","num_warps","num_stages")}
    if not any(all(c[k]==v for k,v in base.items()) for c in CONFIGS.values()): raise ValueError("schedule base outside manifest")
    if not allow_diagnostic and out not in CONFIGS.values(): raise ValueError("public run accepts exactly16 declared configs")
    return out

def _run_impl(x,ql,qh,scale,steps,*,bucket:int,config=None,allow_diagnostic=False):
    p=common.validate_inputs(x,ql,qh,scale,steps,bucket=bucket); c=_resolve(config,bucket,allow_diagnostic); split=c["split_k"]; common.validate_split(split,p.k); out,dest,ss=common.allocate(p,split); kernel=_direct if split==1 else _split; grid=(triton.cdiv(p.n,c["block_n"]),) if split==1 else (triton.cdiv(p.n,c["block_n"]),split)
    kw=dict(N=p.n,K=p.k,stride_xm=x.stride(0),stride_xk=x.stride(1),stride_qln=ql.stride(0),stride_qlk=ql.stride(1),stride_qhn=qh.stride(0),stride_qhk=qh.stride(1),stride_sn=scale.stride(0),stride_sk=scale.stride(1),stride_stn=steps.stride(0),stride_stk=steps.stride(1),stride_om=dest.stride(-2),stride_on=dest.stride(-1),BLOCK_M=bucket,BLOCK_N=c["block_n"],FRAGMENT_K=c["fragment_k"],PIPELINE_STAGES=c["num_stages"],num_warps=c["num_warps"],num_stages=c["num_stages"])
    if split>1: kw.update(stride_os=ss,SPLIT_K=split)
    kernel[grid](x,ql,qh,scale,steps,dest,p.m,**kw); common.finish(dest,out,split); return out

def run(x,ql,qh,scale,steps,*,bucket:int,config=None):
    return _run_impl(x,ql,qh,scale,steps,bucket=bucket,config=config)


def _run_diagnostic(x,ql,qh,scale,steps,*,bucket:int,config):
    """Private off-manifest helper for bounded diagnostics only."""
    return _run_impl(x,ql,qh,scale,steps,bucket=bucket,config=config,allow_diagnostic=True)


_DIRECT_KERNEL=_direct
_SPLIT_KERNEL=_split
__all__=["CONFIGS","run"]
