"""Test: triton_op with runtime dispatch + aot_compile."""
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _mul_kernel(x_ptr, out_ptr, n: tl.constexpr):
    idx = tl.arange(0, n)
    x = tl.load(x_ptr + idx)
    tl.store(out_ptr + idx, x * 2)


@triton.jit
def _add_kernel(x_ptr, out_ptr, n: tl.constexpr):
    idx = tl.arange(0, n)
    x = tl.load(x_ptr + idx)
    tl.store(out_ptr + idx, x + 1)


@triton_op("test::dispatch_triton", mutates_args={})
def dispatch_triton(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    if x.shape[0] == 1:
        wrap_triton(_mul_kernel)[(1,)](x, out, n)
    else:
        wrap_triton(_add_kernel)[(1,)](x, out, n)
    return out


if __name__ == "__main__":
    x1 = torch.randn(1, 4, device="cuda")
    x2 = torch.randn(2, 4, device="cuda")
    print("T=1:", dispatch_triton(x1))
    print("T=2:", dispatch_triton(x2))

    from torch.export import Dim, export

    class M(torch.nn.Module):
        def forward(self, x):
            return torch.ops.test.dispatch_triton(x)

    m = M().cuda()
    seq = Dim("seq", min=1, max=128)
    prog = export(m, (x2,), dynamic_shapes=({0: seq},))
    gm = prog.module()
    print("aot_compile...")
    try:
        paths = torch._inductor.aot_compile(gm, (x2,))
        print(f"SUCCESS: {paths}")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {str(e)[:300]}")
