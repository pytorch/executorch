#!/usr/bin/env python3
"""
Debug test for bf16 linear layer issue.

This test isolates the bf16 matmul bug in the MLX delegate.
"""

import os
import subprocess
import tempfile

import torch
import torch.nn as nn


def test_bf16_linear_minimal():
    """Test bf16 linear with small fixed weights."""

    # Simple 2x2 linear: y = x @ W^T + b
    # x = [1, 2], W = [[0.5, 0.25], [0.125, 0.0625]], b = [0, 0]
    # y[0] = 1*0.5 + 2*0.25 = 0.5 + 0.5 = 1.0
    # y[1] = 1*0.125 + 2*0.0625 = 0.125 + 0.125 = 0.25

    class LinearModel(nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.linear = nn.Linear(2, 2, bias=True)
            # Set known weights
            with torch.no_grad():
                self.linear.weight.copy_(
                    torch.tensor([[0.5, 0.25], [0.125, 0.0625]], dtype=dtype)
                )
                self.linear.bias.copy_(torch.tensor([0.0, 0.0], dtype=dtype))

        def forward(self, x):
            return self.linear(x)

    x_fp32 = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    x_bf16 = x_fp32.to(torch.bfloat16)

    # Test in pure PyTorch first
    model_fp32 = LinearModel(torch.float32)
    model_bf16 = LinearModel(torch.bfloat16).to(torch.bfloat16)

    out_fp32 = model_fp32(x_fp32)
    out_bf16 = model_bf16(x_bf16)

    print(f"PyTorch FP32 output: {out_fp32}")
    print(f"PyTorch BF16 output: {out_bf16}")
    print("Expected: [[1.0, 0.25]]")
    print()

    from executorch.backends.apple.mlx.mlx_preprocess import MLXBackend

    # Now test through MLX delegate
    from executorch.exir import EdgeCompileConfig, to_edge

    for dtype_name, dtype, x_input in [
        ("fp32", torch.float32, x_fp32),
        ("bf16", torch.bfloat16, x_bf16),
    ]:
        print(f"\n=== Testing {dtype_name} through MLX delegate ===")

        model = LinearModel(dtype).to(dtype)
        model.eval()

        x = x_input.clone()

        # Export
        ep = torch.export.export(model, (x,), strict=False)
        edge = to_edge(ep, compile_config=EdgeCompileConfig(_check_ir_validity=False))

        # Delegate to MLX
        lowered = edge.to_backend(MLXBackend())

        # Save
        with tempfile.NamedTemporaryFile(suffix=".pte", delete=False) as f:
            pte_path = f.name

        et_program = lowered.to_executorch()
        with open(pte_path, "wb") as f:
            f.write(et_program.buffer)

        print(f"Saved to {pte_path}")

        # Run with C++ runner
        runner_path = "/Users/scroy/Desktop/executorch/cmake-out/backends/apple/mlx/test/mlx_op_test_runner"

        # Format input for runner
        input_spec = f"f{dtype_name[-2:]}:[1,2]:[{x[0,0].item()},{x[0,1].item()}]"

        cmd = [runner_path, pte_path, input_spec]
        print(f"Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"stdout: {result.stdout}")
        if result.stderr:
            print(f"stderr: {result.stderr}")

        os.unlink(pte_path)


def test_bf16_constant_bytes():
    """Check how bf16 values are serialized."""

    # Create a simple bf16 constant
    val = torch.tensor([0.5], dtype=torch.bfloat16)

    # Get the bytes - bf16 needs special handling since numpy doesn't support it
    val_bytes = val.view(torch.uint16).numpy().tobytes()
    print(f"bf16 value 0.5 bytes: {val_bytes.hex()}")

    # In bf16: 0.5 = 0x3F00 (same as fp32's top 16 bits)
    # Let's verify
    fp32_val = torch.tensor([0.5], dtype=torch.float32)
    fp32_bytes = fp32_val.numpy().tobytes()
    print(f"fp32 value 0.5 bytes: {fp32_bytes.hex()}")

    # bf16 should be the top 2 bytes of fp32
    expected_bf16 = fp32_bytes[2:4]  # Little endian: bytes 2-3 are the high bytes
    print(f"Expected bf16 (from fp32 high bytes): {expected_bf16.hex()}")


def test_matmul_direct_mlx():
    """Verify MLX bf16 matmul works directly."""
    import mlx.core as mx

    # Same values as above
    x = mx.array([[1.0, 2.0]], dtype=mx.bfloat16)
    w = mx.array([[0.5, 0.25], [0.125, 0.0625]], dtype=mx.bfloat16)

    # Linear: y = x @ W^T
    wt = mx.transpose(w, (1, 0))
    y = mx.matmul(x, wt)
    mx.eval(y)

    print(f"Direct MLX bf16 matmul result: {y}")
    print("Expected: [[1.0, 0.25]]")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing bf16 constant bytes")
    print("=" * 60)
    test_bf16_constant_bytes()

    print("\n" + "=" * 60)
    print("Testing direct MLX bf16 matmul")
    print("=" * 60)
    test_matmul_direct_mlx()

    print("\n" + "=" * 60)
    print("Testing bf16 linear through delegate")
    print("=" * 60)
    test_bf16_linear_minimal()
