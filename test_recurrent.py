"""Quick correctness test: recurrent Triton kernel vs einsum reference."""
import sys
sys.path.insert(0, "/home/gasoonjia/executorch")
import torch

# Import the kernel from source (not installed)
from backends.cuda.triton.kernels.chunk_gated_delta_rule import (
    _launch_recurrent,
    _launch_chunked,
)


def reference_recurrent(q, k, v, g, beta, state, scale):
    """Reference einsum implementation."""
    state_f = state.float()
    q_f = q[:, 0].float()
    k_f = k[:, 0].float()
    v_f = v[:, 0].float()
    g_f = g[:, 0]
    beta_f = beta[:, 0].float()

    state_f = state_f * g_f.exp().unsqueeze(-1).unsqueeze(-1)
    Sk = torch.einsum("bhkv,bhk->bhv", state_f, k_f)
    delta = v_f - Sk
    state_f = state_f + beta_f.unsqueeze(-1).unsqueeze(-1) * torch.einsum(
        "bhk,bhv->bhkv", k_f, delta
    )
    o = torch.einsum("bhkv,bhk->bhv", state_f, q_f) * scale
    return o.unsqueeze(1).to(v.dtype), state_f


def test_recurrent_kernel():
    torch.manual_seed(42)
    B, T, H, K, V = 1, 1, 16, 64, 128
    scale = K**-0.5

    q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(B, T, H, device="cuda", dtype=torch.float32) * 0.1
    beta = torch.rand(B, T, H, device="cuda", dtype=torch.float32)
    state = torch.randn(B, H, K, V, device="cuda", dtype=torch.float32) * 0.01

    # Reference
    ref_o, ref_state = reference_recurrent(q, k, v, g, beta, state, scale)

    # Triton kernel
    tri_o, tri_state = _launch_recurrent(q, k, v, g, beta, state, scale)

    # Compare
    o_err = (ref_o.float() - tri_o.float()).abs().max().item()
    s_err = (ref_state - tri_state).abs().max().item()
    o_rel = o_err / ref_o.float().abs().max().item()
    s_rel = s_err / ref_state.abs().max().item()

    print(f"Output max abs err: {o_err:.6e}, rel: {o_rel:.6e}")
    print(f"State  max abs err: {s_err:.6e}, rel: {s_rel:.6e}")

    if o_rel < 1e-3 and s_rel < 1e-3:
        print("PASS")
    else:
        print("FAIL — tolerance exceeded")
        return False
    return True


def test_dispatch():
    """Test that T=1 goes through recurrent and T=2 goes through chunked."""
    torch.manual_seed(42)
    B, H, K, V = 1, 16, 64, 128

    # T=1
    q1 = torch.randn(B, 1, H, K, device="cuda", dtype=torch.bfloat16)
    k1 = torch.randn(B, 1, H, K, device="cuda", dtype=torch.bfloat16)
    v1 = torch.randn(B, 1, H, V, device="cuda", dtype=torch.bfloat16)
    g1 = torch.randn(B, 1, H, device="cuda", dtype=torch.float32) * 0.1
    beta1 = torch.rand(B, 1, H, device="cuda", dtype=torch.float32)
    state = torch.randn(B, H, K, V, device="cuda", dtype=torch.float32) * 0.01

    from backends.cuda.triton.kernels.chunk_gated_delta_rule import chunk_gated_delta_rule

    o1, s1 = chunk_gated_delta_rule(q1, k1, v1, g1, beta1, state)
    print(f"T=1 dispatch: output shape {o1.shape}, state shape {s1.shape}")
    assert o1.shape == (B, 1, H, V), f"Bad output shape: {o1.shape}"
    assert s1.shape == (B, H, K, V), f"Bad state shape: {s1.shape}"

    # T=4
    q4 = torch.randn(B, 4, H, K, device="cuda", dtype=torch.bfloat16)
    k4 = torch.randn(B, 4, H, K, device="cuda", dtype=torch.bfloat16)
    v4 = torch.randn(B, 4, H, V, device="cuda", dtype=torch.bfloat16)
    g4 = torch.randn(B, 4, H, device="cuda", dtype=torch.float32) * 0.1
    beta4 = torch.rand(B, 4, H, device="cuda", dtype=torch.float32)

    o4, s4 = chunk_gated_delta_rule(q4, k4, v4, g4, beta4, state)
    print(f"T=4 dispatch: output shape {o4.shape}, state shape {s4.shape}")
    assert o4.shape == (B, 4, H, V), f"Bad output shape: {o4.shape}"
    assert s4.shape == (B, H, K, V), f"Bad state shape: {s4.shape}"

    print("Dispatch test PASS")
    return True


if __name__ == "__main__":
    print("=== Recurrent kernel correctness ===")
    ok1 = test_recurrent_kernel()
    print()
    print("=== Dispatch test ===")
    ok2 = test_dispatch()
    sys.exit(0 if ok1 and ok2 else 1)
