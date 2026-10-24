# Optimized by KernelAgent-Oink(https://github.com/meta-pytorch/KernelAgent)
"""Strict planar-Q6 contract and deterministic split reduction for W6A16."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import triton
import triton.language as tl

GROUP_SIZE = 16
SUPER_BLOCK = 256
SUPPORTED_SPLITS = (1, 2, 4, 8, 16, 32)

@dataclass(frozen=True)
class Problem:
    m:int; n:int; k:int; bucket:int; device:torch.device


def validate_inputs(x, ql, qh, scale, steps, *, bucket:int) -> Problem:
    if bucket not in (8,16): raise ValueError("bucket must be 8 or 16")
    if any(t.ndim!=2 for t in (x,ql,qh,scale,steps)): raise ValueError("all inputs must be rank2")
    m,k=x.shape; n,k2=ql.shape
    if not 1<=m<=bucket or k<=0 or k%256 or n<=0: raise ValueError("invalid M/N/K")
    if k2*2!=k or tuple(qh.shape)!=(n,k//4): raise ValueError("ql/qh shape mismatch")
    if tuple(scale.shape)!=(n,k//16) or tuple(steps.shape)!=(n,k//256): raise ValueError("scale/step shape mismatch")
    expected=(("x",x,torch.bfloat16),("ql",ql,torch.uint8),("qh",qh,torch.uint8),("scale",scale,torch.int8),("steps",steps,torch.float16))
    for name,t,dtype in expected:
        if t.dtype!=dtype: raise TypeError(f"{name} must be {dtype}, got {t.dtype}")
        if not t.is_contiguous(): raise ValueError(f"{name} must be contiguous")
        if t.device!=x.device: raise ValueError("all inputs must share device")
    if x.device.type!="cuda": raise ValueError("CUDA required")
    return Problem(m,n,k,bucket,x.device)


def validate_split(split:int,k:int) -> None:
    if split not in SUPPORTED_SPLITS: raise ValueError(f"split must be one of {SUPPORTED_SPLITS}")
    if split>k//256: raise ValueError(f"split {split} exceeds K256 blocks {k//256}")


def normalize_config(config:dict[str,Any]|None,defaults:dict[str,Any],allowed:set[str])->dict[str,Any]:
    out=dict(defaults)
    if config is not None:
        unknown=set(config)-allowed
        if unknown: raise ValueError(f"unknown config keys {sorted(unknown)}")
        out.update(config)
    return out


@triton.jit
def _reduce(partial,out,m,N:tl.constexpr,stride_ps:tl.constexpr,stride_pm:tl.constexpr,stride_pn:tl.constexpr,stride_om:tl.constexpr,stride_on:tl.constexpr,BLOCK_M:tl.constexpr,BLOCK_N:tl.constexpr,SPLIT_K:tl.constexpr):
    pn=tl.program_id(0); pm=tl.program_id(1); on=pn*BLOCK_N+tl.arange(0,BLOCK_N); om=pm*BLOCK_M+tl.arange(0,BLOCK_M); mask=(om[:,None]<m)&(on[None,:]<N); acc=tl.zeros((BLOCK_M,BLOCK_N),tl.float32)
    for s in range(SPLIT_K): acc+=tl.load(partial+s*stride_ps+om[:,None]*stride_pm+on[None,:]*stride_pn,mask=mask,other=0.0)
    tl.store(out+om[:,None]*stride_om+on[None,:]*stride_on,acc.to(tl.bfloat16),mask=mask)


def allocate(problem:Problem,split:int):
    out=torch.empty((problem.m,problem.n),dtype=torch.bfloat16,device=problem.device)
    if split==1:return out,out,0
    partial=torch.empty((split,problem.bucket,problem.n),dtype=torch.float32,device=problem.device)
    return out,partial,partial.stride(0)


def finish(partial,out,split:int):
    if split==1:return
    m,n=out.shape; bm=triton.next_power_of_2(m); bn=64
    _reduce[(triton.cdiv(n,bn),triton.cdiv(m,bm))](partial,out,m,N=n,stride_ps=partial.stride(0),stride_pm=partial.stride(1),stride_pn=partial.stride(2),stride_om=out.stride(0),stride_on=out.stride(1),BLOCK_M=bm,BLOCK_N=bn,SPLIT_K=split,num_warps=4,num_stages=1)

REDUCE_KERNEL=_reduce
