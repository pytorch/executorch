"""Validate recurrent gated delta rule vs chunked FLA version."""
import torch
import torch.nn.functional as F

# Import the chunked version
import executorch.backends.cuda.triton.kernels  # register triton ops


def recurrent_gated_delta_rule(q, k, v, g, beta, initial_state):
    """Recurrent gated delta rule (reference implementation)."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    scale = K**-0.5
    state = initial_state.float()

    outputs = []
    for t in range(T):
        q_t = q[:, t].float()
        k_t = k[:, t].float()
        v_t = v[:, t].float()
        g_t = g[:, t]
        beta_t = beta[:, t].float()

        state = state * g_t.exp().unsqueeze(-1).unsqueeze(-1)
        Sk = torch.einsum("bhkv,bhk->bhv", state, k_t)
        delta = v_t - Sk
        state = state + beta_t.unsqueeze(-1).unsqueeze(-1) * torch.einsum(
            "bhk,bhv->bhkv", k_t, delta
        )
        o_t = torch.einsum("bhkv,bhk->bhv", state, q_t) * scale
        outputs.append(o_t)

    output = torch.stack(outputs, dim=1).to(q.dtype)
    return output, state


def test_correctness():
    """Compare recurrent vs chunked for T=1."""
    torch.manual_seed(42)
    B, T, H, K, V = 1, 1, 32, 128, 128

    q = F.normalize(torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16), dim=-1)
    k = F.normalize(torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16), dim=-1)
    v = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(B, T, H, device="cuda", dtype=torch.float32)  # negative log-space
    beta = torch.rand(B, T, H, device="cuda", dtype=torch.bfloat16).sigmoid()
    initial_state = torch.randn(B, H, K, V, device="cuda", dtype=torch.bfloat16) * 0.01

    # Chunked version
    with torch.no_grad():
        o_chunked, s_chunked = torch.ops.triton.chunk_gated_delta_rule(
            q, k, v, g, beta, initial_state
        )

    # Recurrent version
    with torch.no_grad():
        o_recurrent, s_recurrent = recurrent_gated_delta_rule(
            q, k, v, g, beta, initial_state
        )

    # Compare
    o_diff = (o_chunked.float() - o_recurrent.float()).abs().max().item()
    s_diff = (s_chunked.float() - s_recurrent.float()).abs().max().item()

    print(f"T=1 correctness check:")
    print(f"  Output max diff: {o_diff:.6f}")
    print(f"  State  max diff: {s_diff:.6f}")

    # Also test T>1 for completeness
    T2 = 4
    q2 = F.normalize(torch.randn(B, T2, H, K, device="cuda", dtype=torch.bfloat16), dim=-1)
    k2 = F.normalize(torch.randn(B, T2, H, K, device="cuda", dtype=torch.bfloat16), dim=-1)
    v2 = torch.randn(B, T2, H, V, device="cuda", dtype=torch.bfloat16)
    g2 = -torch.rand(B, T2, H, device="cuda", dtype=torch.float32)
    beta2 = torch.rand(B, T2, H, device="cuda", dtype=torch.bfloat16).sigmoid()
    state2 = torch.randn(B, H, K, V, device="cuda", dtype=torch.bfloat16) * 0.01

    with torch.no_grad():
        o_chunked2, s_chunked2 = torch.ops.triton.chunk_gated_delta_rule(
            q2, k2, v2, g2, beta2, state2
        )
        o_recurrent2, s_recurrent2 = recurrent_gated_delta_rule(
            q2, k2, v2, g2, beta2, state2
        )

    o_diff2 = (o_chunked2.float() - o_recurrent2.float()).abs().max().item()
    s_diff2 = (s_chunked2.float() - s_recurrent2.float()).abs().max().item()

    print(f"\nT={T2} correctness check:")
    print(f"  Output max diff: {o_diff2:.6f}")
    print(f"  State  max diff: {s_diff2:.6f}")

    # Relative errors
    o_rel = o_diff / (o_chunked.float().abs().max().item() + 1e-10)
    s_rel = s_diff / (s_chunked.float().abs().max().item() + 1e-10)
    print(f"\nT=1 relative errors: output={o_rel:.6f}, state={s_rel:.6f}")

    passed = o_diff < 0.01 and s_diff < 0.01
    print(f"\n{'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    test_correctness()
