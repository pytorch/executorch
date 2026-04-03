"""Benchmark recurrent vs chunked FLA in full model decode with torch.compile.

Usage:
  # Recurrent (current code):
  python bench_fla.py --prequantized ~/models/Qwen3.5-35B-A3B-HQQ-INT4-local --mode recurrent
  # Chunked (original FLA triton kernels):
  python bench_fla.py --prequantized ~/models/Qwen3.5-35B-A3B-HQQ-INT4-local --mode chunked
"""
import argparse
import time
import torch


def patch_chunked():
    """Restore chunked FLA in GatedDeltaNet before model construction."""
    import executorch.examples.models.qwen3_5_moe.model as mod

    original_forward = mod.GatedDeltaNet.forward

    def chunked_forward(self, x, input_pos):
        """GatedDeltaNet.forward using chunked FLA triton kernels."""
        import torch.nn.functional as F

        B, T, _ = x.size()

        reset = (input_pos[0] == 0).to(self.conv_state.dtype)
        keep = 1.0 - reset
        self.conv_state[:B].mul_(keep)
        self.recurrent_state[:B].mul_(keep)

        proj = self.in_proj(x)
        cd = self.conv_dim
        vd = self.value_dim
        nh = self.num_v_heads
        mixed_qkv = proj[..., :cd]
        z = proj[..., cd : cd + vd].reshape(B, T, self.num_v_heads, self.head_v_dim)
        b = proj[..., cd + vd : cd + vd + nh]
        a = proj[..., cd + vd + nh :]

        qkv_t = mixed_qkv.transpose(1, 2)
        conv_input = torch.cat([self.conv_state[:B], qkv_t], dim=-1)
        with torch.no_grad():
            self.conv_state[:B].copy_(conv_input[:, :, -self.conv_kernel_size :])
        w = self.conv1d.weight.squeeze(1).float()
        T_conv = conv_input.shape[-1] - self.conv_kernel_size + 1
        acc = torch.zeros(
            B, conv_input.shape[1], T_conv,
            dtype=torch.float32, device=conv_input.device,
        )
        for k in range(self.conv_kernel_size):
            acc = acc + conv_input[:, :, k : k + T_conv].float() * w[:, k : k + 1]
        qkv_conv = F.silu(acc[:, :, -T:]).to(conv_input.dtype).transpose(1, 2)

        kd = self.key_dim
        q = qkv_conv[..., :kd].reshape(B, T, self.num_k_heads, self.head_k_dim)
        k = qkv_conv[..., kd : 2 * kd].reshape(B, T, self.num_k_heads, self.head_k_dim)
        v = qkv_conv[..., 2 * kd :].reshape(B, T, self.num_v_heads, self.head_v_dim)

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        if self.head_repeat > 1:
            q = q.repeat_interleave(self.head_repeat, dim=2)
            k = k.repeat_interleave(self.head_repeat, dim=2)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # Use chunked FLA triton kernels
        output, state = torch.ops.triton.chunk_gated_delta_rule(
            q, k, v, g, beta, self.recurrent_state[:B]
        )
        with torch.no_grad():
            self.recurrent_state[:B].copy_(state)

        output = output.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        output = self.norm(output, z)
        output = output.reshape(B, T, -1)

        return self.out_proj(output)

    mod.GatedDeltaNet.forward = chunked_forward
    print("Patched: using chunked FLA triton kernels")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prequantized", required=True)
    parser.add_argument("--mode", choices=["recurrent", "chunked"], required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    # Patch BEFORE any model import if chunked
    if args.mode == "chunked":
        patch_chunked()

    import executorch.backends.cuda.triton.kernels  # register triton ops
    from executorch.examples.models.qwen3_5_moe.export import load_prequantized_model
    from executorch.examples.models.qwen3_5_moe.inference import _move_to_cuda

    print("Loading model...")
    model, config = load_prequantized_model(args.prequantized, max_seq_len=4096)
    _move_to_cuda(model, config)
    model.eval()

    if not args.no_compile:
        print("Compiling with torch.compile...")
        model = torch.compile(model, mode="default")

    # Warmup
    print(f"Warming up ({args.warmup} steps)...")
    with torch.no_grad():
        for i in range(args.warmup):
            tok = torch.tensor([[1]], dtype=torch.long, device="cuda")
            pos = torch.tensor([i], dtype=torch.long, device="cuda")
            model(tok, pos)
    torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({args.steps} decode steps)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(args.steps):
            tok = torch.tensor([[1]], dtype=torch.long, device="cuda")
            pos = torch.tensor([args.warmup + i], dtype=torch.long, device="cuda")
            model(tok, pos)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tok_s = args.steps / elapsed
    ms_per_step = elapsed / args.steps * 1000
    print(f"\nResult [{args.mode}]: {tok_s:.1f} tok/s ({ms_per_step:.2f} ms/step, {args.steps} steps)")


if __name__ == "__main__":
    main()
