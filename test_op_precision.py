#!/usr/bin/env python3
"""Pinpoint: which TRT operation(s) cause the most precision loss?
Test individual ops in isolation: matmul, softmax, layernorm, conv1d."""
import sys
import os
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/gasoonjia/trt/executorch/examples/models/parakeet")
from executorch.exir import to_edge
from executorch.backends.nvidia.tensorrt.partitioner import TensorRTPartitioner
from executorch.runtime import Runtime

runtime = Runtime.get()

def test_trt_precision(model, name, *inputs):
    """Export model, run through TRT, compare to eager."""
    model.eval()
    with torch.no_grad():
        eager_out = model(*inputs)
    if isinstance(eager_out, tuple):
        eager_main = eager_out[0]
    else:
        eager_main = eager_out

    try:
        # Build kwargs for export if needed
        ep = torch.export.export(model, inputs)
        edge = to_edge(ep)
        edge = edge.to_backend(TensorRTPartitioner())
        et_prog = edge.to_executorch()

        tmp_path = f"/tmp/test_op_{name}.pte"
        with open(tmp_path, "wb") as f:
            f.write(et_prog.buffer)

        program = runtime.load_program(tmp_path)
        method = program.load_method("forward")
        trt_out = method.execute(list(inputs))
        if isinstance(trt_out, (list, tuple)):
            trt_main = trt_out[0]
        else:
            trt_main = trt_out

        eager_np = eager_main.detach().numpy()
        trt_np = trt_main.detach().numpy()

        cos = np.dot(eager_np.flatten(), trt_np.flatten()) / (
            np.linalg.norm(eager_np.flatten()) * np.linalg.norm(trt_np.flatten()) + 1e-8
        )
        max_diff = np.abs(eager_np - trt_np).max()
        mean_diff = np.abs(eager_np - trt_np).mean()
        rel_diff = (np.abs(eager_np - trt_np) / (np.abs(eager_np) + 1e-8)).mean()

        os.unlink(tmp_path)
        print(f"  {name:40s} cos={cos:.8f}  max_diff={max_diff:.6f}  mean_diff={mean_diff:.8f}  rel_diff={rel_diff:.6f}")
        return cos, max_diff
    except Exception as e:
        print(f"  {name:40s} FAILED: {e}")
        import traceback; traceback.print_exc()
        return None, None


# Generate realistic attention-like inputs
B, H, T, D = 1, 16, 93, 64  # Matching Parakeet's attention dims
torch.manual_seed(42)

# 1. MatMul (Q*K^T)
class MatMulQK(nn.Module):
    def forward(self, q, k):
        return torch.matmul(q, k.transpose(-2, -1))

q = torch.randn(B, H, T, D)
k = torch.randn(B, H, T, D)

# 2. Softmax
class SoftmaxModel(nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1)

# Real attention scores (scaled)
scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)

# 3. BMM (attention * V)
class BMMModel(nn.Module):
    def forward(self, attn, v):
        return torch.matmul(attn, v)

attn_weights = torch.softmax(scores, dim=-1)
v = torch.randn(B, H, T, D)

# 4. LayerNorm
class LayerNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(1024)
    def forward(self, x):
        return self.ln(x)

ln_model = LayerNormModel()
# Random init weights
ln_input = torch.randn(B, T, 1024)

# 5. Conv1d + BatchNorm
class Conv1dBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1024, 1024, kernel_size=31, padding=15, groups=1024)
        self.bn = nn.BatchNorm1d(1024)
    def forward(self, x):
        return self.bn(self.conv(x))

conv_model = Conv1dBN()
conv_model.eval()
conv_input = torch.randn(B, 1024, T)

# 6. Linear (feed-forward)
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1024, 4096)
    def forward(self, x):
        return self.linear(x)

linear_model = LinearModel()
linear_input = torch.randn(B, T, 1024)

# 7. Full attention: Q*K^T -> softmax -> V matmul
class FullAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = D ** -0.5
    def forward(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

# 8. Full attention with masked_fill (-10000.0) — simulating real mask behavior with all-False mask
class FullAttentionMasked(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = D ** -0.5
    def forward(self, q, k, v, mask):
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(mask, -10000.0)
        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(mask, 0.0)
        return torch.matmul(attn, v)

# All-false mask (no padding)
mask_false = torch.zeros(B, 1, T, T, dtype=torch.bool)

# 9. Full attention with masked_fill and SOME masked positions
class FullAttentionPartialMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = D ** -0.5
    def forward(self, q, k, v, mask):
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(mask, -10000.0)
        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(mask, 0.0)
        return torch.matmul(attn, v)

# Partial mask: mask last 20 positions (simulating padding)
mask_partial = torch.zeros(B, 1, T, T, dtype=torch.bool)
mask_partial[:, :, :, 73:] = True  # Last 20 cols masked
mask_partial[:, :, 73:, :] = True  # Last 20 rows masked

# 10. Softmax with pre-masked scores (some positions -10000)
class SoftmaxMasked(nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1)

masked_scores = scores.clone()
masked_scores[:, :, :, 73:] = -10000.0


print(f"\n{'='*90}")
print("TRT OPERATION PRECISION TEST")
print(f"{'='*90}")
print(f"  Shapes: B={B}, H={H}, T={T}, D={D}")
print(f"  {'Op':<40s} {'cosine':>10s}  {'max_diff':>10s}  {'mean_diff':>12s}  {'rel_diff':>10s}")
print(f"  {'-'*86}")

test_trt_precision(MatMulQK(), "matmul_QKt", q, k)
test_trt_precision(SoftmaxModel(), "softmax_real_scores", scores)
test_trt_precision(SoftmaxModel(), "softmax_masked_scores", masked_scores)
test_trt_precision(BMMModel(), "bmm_attn_x_V", attn_weights, v)
test_trt_precision(LayerNormModel(), "layernorm_1024", ln_input)
test_trt_precision(Conv1dBN(), "conv1d_bn_k31", conv_input)
test_trt_precision(LinearModel(), "linear_1024x4096", linear_input)
test_trt_precision(FullAttention(), "full_attention_no_mask", q, k, v)
test_trt_precision(FullAttentionMasked(), "full_attention_allFalse_mask", q, k, v, mask_false)
test_trt_precision(FullAttentionPartialMask(), "full_attention_partial_mask", q, k, v, mask_partial)
