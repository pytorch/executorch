"""Measured large-M native W4A16 launches inside the existing bulk forward.

The policy is supplied by a separate, verified same-input operator report.
Original static methods and unsupported Q4 shapes retain their operations.
Keep the conditional at a Triton operator boundary so both branches lower
through Inductor without introducing portable dequantization operations.
"""
import hashlib
from pathlib import Path

import torch
import triton
from torch.library import triton_op
from executorch.backends.cuda.quantize_op_dispatch.int4_dispatch import (
    _dequant_matmul as original_dequant_matmul,
)
from executorch.backends.cuda.triton.kernels import int4_quantized_gemm as native


POLICIES = []
DISPATCH_SHA256 = '1a3fdadd5dcfe03ef9b6f55f93e4fce9a88b44867839b1316c824f7bd1faaa3e'
NATIVE_SHA256 = 'fb72d31e6d3f91e076fe9721744eacb7c379ffe0612c57e3c48ae4a25c20027c'


def register_policy(policy):
    """Freeze an explicit ordered list of measured row limits and launches."""
    assert hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() == NATIVE_SHA256
    if policy.get('actual_row_grid', False):
        from roofline_large_row_grid import install
        install()
    shape = tuple(policy['shape'])
    assert len(shape) == 2 and all(isinstance(v, int) and v > 0 for v in shape)
    branches = []
    previous = 64
    for item in policy['branches']:
        limit, bucket, split = item['max_M'], item['bucket'], item['split_k']
        assert previous < limit <= bucket and bucket in (128, 256, 512, 1024, 2048, 4096)
        assert split in native.TILE_SPLIT_K_CANDIDATES
        c = item['config']
        config = triton.Config(dict(c['kwargs']), num_warps=c['num_warps'], num_stages=c['num_stages'])
        branches.append((limit, bucket, split, config))
        previous = limit
    assert branches
    POLICIES.append((shape, tuple(branches)))
    return len(POLICIES) - 1


def _dispatch_branch(args, branches, index):
    if index == len(branches):
        return original_dequant_matmul(*args, 32)
    limit, bucket, split, config = branches[index]
    return torch.cond(args[0].shape[0] <= limit,
        lambda a,b,c,d,e,f: native._launch_w4a16(
            bucket, a,b,c,d,e,f,32, split_k=split, config=config),
        lambda a,b,c,d,e,f: _dispatch_branch((a,b,c,d,e,f), branches, index+1), args)


@triton_op('siliconbench::roofline_native_large_w4a16', mutates_args={})
def large_w4a16_op(x: torch.Tensor, q: torch.Tensor, sc: torch.Tensor,
                   ss: torch.Tensor, z: torch.Tensor, zs: torch.Tensor,
                   policy_index: int) -> torch.Tensor:
    shape, branches = POLICIES[policy_index]
    assert (q.shape[0], q.shape[1]*2) == shape and x.shape[1] == shape[1]
    return _dispatch_branch((x,q,sc,ss,z,zs), branches, 0)


@large_w4a16_op.register_fake
def _fake(x,q,sc,ss,z,zs,policy_index):
    shape, _ = POLICIES[policy_index]
    assert (q.shape[0], q.shape[1]*2) == shape and x.shape[1] == shape[1]
    assert x.dtype == torch.bfloat16
    return torch.empty((x.shape[0], q.shape[0]), device=x.device, dtype=x.dtype)


@triton_op('siliconbench::roofline_native_large_w4a16_leaf', mutates_args={})
def native_leaf_op(x: torch.Tensor, q: torch.Tensor, sc: torch.Tensor,
                   ss: torch.Tensor, z: torch.Tensor, zs: torch.Tensor,
                   policy_index: int, branch_index: int) -> torch.Tensor:
    """One measured native launch, with control flow exported separately."""
    _, branches = POLICIES[policy_index]
    _, bucket, split, config = branches[branch_index]
    return native._launch_w4a16(bucket,x,q,sc,ss,z,zs,32,split_k=split,config=config)


@native_leaf_op.register_fake
def _native_leaf_fake(x,q,sc,ss,z,zs,policy_index,branch_index):
    return _fake(x,q,sc,ss,z,zs,policy_index)


def large_w4a16_route(x,q,sc,ss,z,zs,policy_index):
    """Export each row predicate before lowering the native Triton leaf.

    Nested torch.cond inside a decomposed triton_op can specialize direct
    branches to an example hint. Explicit outer control flow preserves the
    exported row symbol and lets each leaf lower independently.
    """
    _, branches=POLICIES[policy_index]
    def choose(args,index):
        if index==len(branches):return original_dequant_matmul(*args,32)
        return torch.cond(args[0].shape[0]<=branches[index][0],
            lambda a,b,c,d,e,f:native_leaf_op(a,b,c,d,e,f,policy_index,index),
            lambda a,b,c,d,e,f:choose((a,b,c,d,e,f),index+1),args)
    return choose((x,q,sc,ss,z,zs),0)


def install_export_hook(dispatch, *, capability, policies):
    assert tuple(capability) == (12, 0)
    assert hashlib.sha256(Path(dispatch.__file__).read_bytes()).hexdigest() == DISPATCH_SHA256
    from executorch.backends.cuda.quantize_op_dispatch._gemm_family_dispatch import can_launch_triton
    original = dispatch._dequant_matmul
    assert original is original_dequant_matmul, 'Never stack competing bulk routing hooks'
    indices = {}
    for policy in policies:
        shape = tuple(policy['shape']); assert shape not in indices
        indices[shape] = register_policy(policy)

    def route(x,q,sc,ss,z,zs,group_size):
        shape = (q.shape[0], q.shape[1]*2)
        eager = not torch.compiler.is_compiling() and isinstance(x.shape[0],int)
        CUDA_trace = torch.compiler.is_compiling() or can_launch_triton(x)
        if eager or not CUDA_trace or group_size != 32 or x.dtype != torch.bfloat16 or shape not in indices:
            return original(x,q,sc,ss,z,zs,group_size)
        return large_w4a16_route(x,q,sc,ss,z,zs,indices[shape])

    dispatch._dequant_matmul = route
    return dict(policies=policies, installed=True, original_static_methods_retained=True,
                original_fallback_retained_above_measured_limits=True,
                native_Triton_body_SHA256=NATIVE_SHA256, activation='BF16',
                weight_encoding_changed=False, new_forward_methods=False,
                full_forward_correctness_and_performance_pending=True)
