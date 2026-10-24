# Optimized by KernelAgent-Oink(https://github.com/meta-pytorch/KernelAgent)
"""Pending-SM120 v9 portfolio metadata with unconditional production fallback."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from .kernels import CONFIGS
from .aoti import candidate_op,config_index

VALIDATED_CONTROL = {(8,2048,3840): {"config_id":"v9_m8_attn_control_sp8_r32","source_generation":"v8 actual AOTI","ratio":0.8775510330806879}}
PENDING_TARGETS = {
    (8,3840,15360): ("v9_m8_down_native_sp8_r32","v9_m8_down_native_sp4_r32"),
    (16,3840,15360): ("v9_m16_down_native_sp8_r64","v9_m16_down_native_sp4_r64"),
    (16,30720,3840): ("v9_m16_fused_native_sp1",),
    (8,30720,3840): ("v9_m8_fused_native_sp1",),
    (8,19968,6656): ("v9_m8_mg_atomic_native_sp2",),
    (8,39936,6656): ("v9_m8_mg_fused_native_sp2",),
    (16,19968,6656): ("v9_m16_mg_native_sp1",),
    (16,39936,6656): ("v9_m16_mg_native_sp1",),
}

@dataclass(frozen=True)
class Decision:
    use_candidate: bool
    reason: str
    pending_config_ids: tuple[str,...]=()

def select_policy(*,bucket:int,n:int,k:int,capability:tuple[int,int])->Decision:
    pending=PENDING_TARGETS.get((bucket,n,k),()) if capability==(12,0) else ()
    return Decision(False,"native fallback: v9 routes require fresh actual-AOTI SM120 acceptance",pending)

@dataclass
class NativeBoundLauncher:
    native_launch:Callable
    def __call__(self,*tensors):return self.native_launch(*tensors)

def bind_native(*,native_launch:Callable)->NativeBoundLauncher:
    if not callable(native_launch):raise TypeError("native_launch must be callable")
    return NativeBoundLauncher(native_launch)

def run_pending(config_id:str,x,qdata,scale,scale_step,zero,zero_step):
    if config_id not in CONFIGS:raise ValueError(f"unknown v9 config {config_id!r}")
    return candidate_op(x,qdata,scale,scale_step,zero,zero_step,config_index(config_id))

__all__=["VALIDATED_CONTROL","PENDING_TARGETS","Decision","select_policy","bind_native","run_pending"]
