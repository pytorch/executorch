# Temporary diagnostic wrapper for the Qwen3.5 CUDA export illegal memory access.
# Usage: python .ci/scripts/diag_run_module.py <module> [args...]
import runpy
import sys
import traceback

import torch
from torch._inductor.runtime.triton_heuristics import CachingAutotuner


def _required_nbytes(t):
    if t.numel() == 0:
        return 0
    last = t.storage_offset()
    for size, stride in zip(t.size(), t.stride()):
        last += (size - 1) * stride
    return (last + 1) * t.element_size()


_orig_run = CachingAutotuner.run


def _logged_run(self, *args, **kwargs):
    name = self.inductor_meta.get("kernel_name") or getattr(self.fn, "__name__", "?")
    bad = []
    for i, a in enumerate((*args, *kwargs.values())):
        if isinstance(a, torch.Tensor) and a.is_cuda:
            have = a.untyped_storage().nbytes()
            need = _required_nbytes(a)
            if have < need:
                bad.append(f"arg{i} shape={tuple(a.shape)} storage={have} need={need}")
    msg = f"[diag] autotuner.run {name} launchers={len(self.launchers)}"
    if bad:
        msg += " UNDERSIZED: " + "; ".join(bad)
    print(msg, file=sys.stderr, flush=True)
    return _orig_run(self, *args, **kwargs)


CachingAutotuner.run = _logged_run


def _print_chain(exc):
    seen = set()
    depth = 0
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        print(f"[diag] ===== exception chain depth {depth}: {type(exc).__name__}", file=sys.stderr)
        traceback.print_exception(type(exc), exc, exc.__traceback__, chain=False)
        inner = getattr(exc, "inner_exception", None)
        if inner is not None and id(inner) not in seen:
            exc = inner
        else:
            exc = exc.__cause__ or exc.__context__
        depth += 1


module = sys.argv[1]
sys.argv = [module] + sys.argv[2:]
try:
    runpy.run_module(module, run_name="__main__", alter_sys=True)
except BaseException as e:
    if isinstance(e, SystemExit) and not e.code:
        raise
    _print_chain(e)
    sys.stderr.flush()
    sys.exit(1)
