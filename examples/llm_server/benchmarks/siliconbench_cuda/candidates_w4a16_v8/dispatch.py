# Optimized by KernelAgent-Oink(https://github.com/meta-pytorch/KernelAgent)
"""Pre-trace role/device policy for measured W4A16 schedules and native fallback."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import torch
from .kernels import CONFIGS, run
from .aoti import candidate_op, config_index

SM120 = (12, 0)
ACCEPTED_MAX_RATIO = 0.98
ACCEPTED_POLICY = {
    (8, 2048, 3840): ("v8_m8_attn_down_sp8_r32", "Gemma attention M8", 0.8243478337892932),
    (16, 2048, 3840): ("v8_m16_attn_sp8_r32", "Gemma attention M16", 0.8932495950192884),
    (8, 3840, 15360): ("v8_m8_attn_down_sp8_r32", "Gemma down M8", 0.9433126479315795),
    (16, 30720, 3840): ("v8_m16_fused_sp1", "Gemma fused gate/up M16", 0.9704259776773),
}
PENDING_SYNTHETIC_POLICY = {
    (8, 19968, 6656): ("v8_m8_mg_sp1_pending", "MG synthetic M8 atomic", 0.9455195022345746),
    (8, 39936, 6656): ("v8_m8_mg_sp1_pending", "MG synthetic M8 fused", 0.9734851358410327),
}
if any(ratio >= ACCEPTED_MAX_RATIO for _, _, ratio in ACCEPTED_POLICY.values()):
    raise RuntimeError("accepted policy contains a non-material route")

@dataclass(frozen=True)
class Decision:
    use_candidate: bool
    config_id: str | None
    reason: str
    measured_ratio: float | None = None
    pending_synthetic: bool = False

def select_policy(*, bucket: int, n: int, k: int, capability: tuple[int, int], allow_pending_synthetic: bool = False) -> Decision:
    if capability != SM120:
        return Decision(False, None, f"native fallback: capability {capability} is not SM120")
    key = (bucket, n, k)
    if key in ACCEPTED_POLICY:
        config_id, role, ratio = ACCEPTED_POLICY[key]
        return Decision(True, config_id, f"accepted validated v7 route: {role}", ratio)
    if allow_pending_synthetic and key in PENDING_SYNTHETIC_POLICY:
        config_id, role, ratio = PENDING_SYNTHETIC_POLICY[key]
        return Decision(True, config_id, f"pending synthetic-only route: {role}", ratio, True)
    return Decision(False, None, "native fallback: shape is not an accepted actual-GGUF route")

@dataclass
class BoundLauncher:
    bucket: int
    n: int
    k: int
    device: torch.device
    decision: Decision
    native_launch: Callable
    traceable: bool = True

    @property
    def config_id(self) -> str | None:
        return self.decision.config_id

    def _candidate_contract(self, tensors) -> bool:
        x, qdata, scale, scale_step, zero, zero_step = tensors
        if any(t.device != self.device or not t.is_contiguous() for t in tensors):
            return False
        min_m = 5 if self.bucket == 8 else 9
        if (x.ndim != 2 or qdata.ndim != 2 or not min_m <= x.shape[0] <= self.bucket
                or x.shape[1] != self.k or qdata.shape != (self.n, self.k // 2)):
            return False
        return (x.dtype == torch.bfloat16 and qdata.dtype == torch.uint8
                and scale.dtype == zero.dtype == torch.uint8
                and scale_step.dtype == zero_step.dtype == torch.float16
                and scale.shape == zero.shape == (self.n, self.k // 32)
                and scale_step.shape == zero_step.shape == (self.n, self.k // 256))

    def __call__(self, x, qdata, scale, scale_step, zero, zero_step):
        tensors = (x, qdata, scale, scale_step, zero, zero_step)
        if self.decision.use_candidate and self._candidate_contract(tensors):
            if self.traceable:
                return candidate_op(*tensors, config_index(self.decision.config_id))
            return run(*tensors, bucket=self.bucket, config=self.decision.config_id)
        return self.native_launch(*tensors)

def bind_launcher(*, native_launch: Callable, bucket: int, n: int, k: int,
                  device: torch.device, allow_pending_synthetic: bool = False,
                  traceable: bool = True) -> BoundLauncher:
    if not callable(native_launch):
        raise TypeError("native_launch must be callable")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("device must be CUDA")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    capability = torch.cuda.get_device_capability(device)
    decision = select_policy(bucket=bucket, n=n, k=k, capability=capability,
                             allow_pending_synthetic=allow_pending_synthetic)
    return BoundLauncher(bucket, n, k, device, decision, native_launch, traceable)

__all__ = ["ACCEPTED_MAX_RATIO", "ACCEPTED_POLICY", "PENDING_SYNTHETIC_POLICY", "Decision", "BoundLauncher", "select_policy", "bind_launcher"]
