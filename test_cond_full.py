"""Test: torch.cond through ExecuTorch CUDA pipeline - matching actual model pattern."""
import torch
import torch.nn as nn
from torch.export import Dim, export

from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("state", torch.zeros(1, 4, 8, 8))

    def forward(self, x, input_pos):
        B, T, H, K = x.shape
        V = K
        scale = K**-0.5

        def _recurrent(x, state):
            s = state.float()
            x_f = x[:, 0].float()
            s = s + torch.einsum("bhk,bhv->bhkv", x_f, x_f)
            o = torch.einsum("bhkv,bhk->bhv", s, x_f) * scale
            out = o.unsqueeze(1).expand(-1, x.shape[1], -1, -1).to(x.dtype)
            return out.contiguous(), s.contiguous()

        def _chunked(x, state):
            # Simulates a non-aliasing computation
            out = (x * 0.5 + x * 0.5)  # non-aliasing
            new_state = state + torch.einsum(
                "bhk,bhv->bhkv", x[:, -1].float(), x[:, -1].float()
            )
            return out.contiguous(), new_state.float().contiguous()

        is_single = input_pos[0] == input_pos[-1]
        output, new_state = torch.cond(
            is_single, _recurrent, _chunked, (x, self.state[:B])
        )

        with torch.no_grad():
            self.state[:B].copy_(new_state)

        return output


def main():
    m = M().cuda()
    x = torch.randn(1, 2, 4, 8, device="cuda")
    input_pos = torch.tensor([0, 1], device="cuda")

    seq = Dim("seq", min=1, max=128)
    print("Step 1: Exporting...")
    prog = export(m, (x, input_pos), dynamic_shapes=({1: seq}, {0: seq}), strict=True)
    print("Export OK")

    # Test aot_compile directly (no ExecuTorch)
    print("Step 2: Direct aot_compile...")
    gm = prog.module()
    try:
        paths = torch._inductor.aot_compile(gm, (x, input_pos))
        print(f"Direct aot_compile: SUCCESS")
    except Exception as e:
        print(f"Direct aot_compile: FAILED - {type(e).__name__}: {str(e)[:200]}")

    # Test through ExecuTorch
    CudaBackend.get_decomposition_table = classmethod(lambda cls: {})
    compile_specs = [CudaBackend.generate_method_name_compile_spec("forward")]
    print("Step 3: ExecuTorch pipeline...")
    try:
        et_prog = to_edge_transform_and_lower(
            prog,
            partitioner=[CudaPartitioner(compile_specs)],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False, _skip_dim_order=True
            ),
        )
        print("ExecuTorch: SUCCESS")
    except Exception as e:
        print(f"ExecuTorch: FAILED - {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
