"""Diagnostic control: conditional boundaries, entirely original ET GEMM.

Match the prepared candidate's exact-size branch structure while executing
the original weight reconstruction and linear operation on every path.
This is a numerical attribution control, never an optimized kernel.
"""
import torch
from torch.library import triton_op
from executorch.backends.cuda.quantize_op_dispatch.int4_dispatch import _dequant_matmul


@triton_op('siliconbench::public_native_branch_gemm', mutates_args={})
def native_branch_op(x: torch.Tensor, q: torch.Tensor, sc: torch.Tensor,
                     ss: torch.Tensor, z: torch.Tensor, zs: torch.Tensor) -> torch.Tensor:
    return _dequant_matmul(x, q, sc, ss, z, zs, 32)


@native_branch_op.register_fake
def _branch_fake(x, q, sc, ss, z, zs):
    return torch.empty((x.shape[0], q.shape[0]), device=x.device, dtype=x.dtype)


@triton_op('siliconbench::public_native_prefill_control', mutates_args={})
def native_prefill_control(x: torch.Tensor, q: torch.Tensor, sc: torch.Tensor,
                           ss: torch.Tensor, z: torch.Tensor, zs: torch.Tensor) -> torch.Tensor:
    return torch.cond(x.shape[0] == 512,
        lambda a,b,c,d,e,f: native_branch_op(a,b,c,d,e,f),
        lambda a,b,c,d,e,f: torch.cond(a.shape[0] == 520,
            lambda g,h,i,j,l,p: native_branch_op(g,h,i,j,l,p),
            lambda g,h,i,j,l,p: _dequant_matmul(g,h,i,j,l,p,32),
            (a,b,c,d,e,f)),
        (x,q,sc,ss,z,zs))


@native_prefill_control.register_fake
def _control_fake(x, q, sc, ss, z, zs):
    return torch.empty((x.shape[0], q.shape[0]), device=x.device, dtype=x.dtype)
