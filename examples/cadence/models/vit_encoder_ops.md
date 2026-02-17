# Vision Transformer (ViT) Encoder - Low-Level Operations

This document details the low-level tensor operations used in the `vit_b_16` encoder from torchvision.

## Model Configuration (ViT-Base/16)

| Parameter | Value |
|-----------|-------|
| Patch Size | 16×16 |
| Hidden Dimension | 768 |
| MLP Dimension | 3072 |
| Number of Heads | 12 |
| Number of Layers | 12 |
| Sequence Length | 197 (196 patches + 1 class token) |
| Head Dimension | 64 (768 / 12) |

---

## Encoder Architecture Overview

```
Input (N, 197, 768)
    │
    ├─── ×12 EncoderBlocks ───┐
    │                         │
    │  ┌──────────────────────┴──────────────────────┐
    │  │ LayerNorm (ln_1)                            │
    │  │     ↓                                       │
    │  │ MultiheadAttention (self_attention)         │
    │  │     ↓                                       │
    │  │ Dropout + Residual Add                      │
    │  │     ↓                                       │
    │  │ LayerNorm (ln_2)                            │
    │  │     ↓                                       │
    │  │ MLPBlock (Linear → GELU → Linear)           │
    │  │     ↓                                       │
    │  │ Residual Add                                │
    │  └─────────────────────────────────────────────┘
    │
    ↓
Final LayerNorm (encoder.ln)
    ↓
Output (N, 197, 768)
```

---

## Detailed Operation Breakdown

### 1. LayerNorm (`ln_1`, `ln_2`, `encoder.ln`)

Normalizes across the hidden dimension (768) with learnable γ (weight) and β (bias).

**Formula:** `y = (x - mean) / sqrt(variance + ε) * γ + β`

| Step | Operation | Input Shape | Output Shape | Description |
|------|-----------|-------------|--------------|-------------|
| 1 | `mean` | (N, 197, 768) | (N, 197, 1) | Compute mean across dim=-1 |
| 2 | `sub` | (N, 197, 768) | (N, 197, 768) | x - mean |
| 3 | `pow` / `mul` | (N, 197, 768) | (N, 197, 768) | Square for variance |
| 4 | `mean` | (N, 197, 768) | (N, 197, 1) | Compute variance |
| 5 | `add` | (N, 197, 1) | (N, 197, 1) | Add ε (1e-6) for numerical stability |
| 6 | `rsqrt` | (N, 197, 1) | (N, 197, 1) | 1 / sqrt(variance + ε) |
| 7 | `mul` | (N, 197, 768) | (N, 197, 768) | Multiply normalized value |
| 8 | `mul` | (N, 197, 768) | (N, 197, 768) | Scale by γ (learned weight) |
| 9 | `add` | (N, 197, 768) | (N, 197, 768) | Add β (learned bias) |

**Parameters:**
- γ (weight): (768,)
- β (bias): (768,)

---

### 2. MultiheadAttention (`self_attention`)

12 attention heads, each with head_dim = 64.

#### 2.1 Input Projection (Q, K, V)

Projects input to Query, Key, and Value vectors using packed weights.

| Step | Operation | Input Shape | Output Shape | Description |
|------|-----------|-------------|--------------|-------------|
| 1 | `linear` | (N, 197, 768) | (N, 197, 2304) | in_proj_weight (2304×768) × x + in_proj_bias (2304) |
| 2 | `unflatten` | (N, 197, 2304) | (N, 197, 3, 768) | Split into Q, K, V |
| 3 | `transpose` | (N, 197, 3, 768) | (3, N, 197, 768) | Reorder dimensions |
| 4 | `contiguous` | - | - | Ensure memory contiguity |
| 5 | `index [0,1,2]` | (3, N, 197, 768) | 3×(N, 197, 768) | Extract Q, K, V |

**Parameters:**
- in_proj_weight: (2304, 768) - packed [W_q, W_k, W_v]
- in_proj_bias: (2304,) - packed [b_q, b_k, b_v]

#### 2.2 Reshape for Multi-Head

| Step | Operation | Input Shape | Output Shape | Description |
|------|-----------|-------------|--------------|-------------|
| 1 | `view` | (N, 197, 768) | (197, N×12, 64) | Reshape Q |
| 2 | `transpose` | (197, N×12, 64) | (N×12, 197, 64) | Batch first |
| 3 | `view` | (N, 197, 768) | (197, N×12, 64) | Reshape K |
| 4 | `transpose` | (197, N×12, 64) | (N×12, 197, 64) | Batch first |
| 5 | `view` | (N, 197, 768) | (197, N×12, 64) | Reshape V |
| 6 | `transpose` | (197, N×12, 64) | (N×12, 197, 64) | Batch first |

#### 2.3 Scaled Dot-Product Attention

**Formula:** `Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V`

**Option A: Standard Path** (when `need_weights=True`)

| Step | Operation | Input Shape | Output Shape | Description |
|------|-----------|-------------|--------------|-------------|
| 1 | `mul` | (N×12, 197, 64) | (N×12, 197, 64) | Scale Q by 1/√64 = 0.125 |
| 2 | `transpose` | (N×12, 197, 64) | (N×12, 64, 197) | Transpose K |
| 3 | `bmm` | Q:(N×12,197,64), K^T:(N×12,64,197) | (N×12, 197, 197) | Batch matrix multiply |
| 4 | `softmax` | (N×12, 197, 197) | (N×12, 197, 197) | Normalize attention scores |
| 5 | `dropout` | (N×12, 197, 197) | (N×12, 197, 197) | Regularization (training only) |
| 6 | `bmm` | Attn:(N×12,197,197), V:(N×12,197,64) | (N×12, 197, 64) | Apply attention to values |

**Option B: Fused SDPA** (when `need_weights=False`)

| Step | Operation | Input Shape | Output Shape | Description |
|------|-----------|-------------|--------------|-------------|
| 1 | `view` | (N×12, 197, 64) | (N, 12, 197, 64) | Reshape Q, K, V |
| 2 | `scaled_dot_product_attention` | Q,K,V:(N,12,197,64) | (N, 12, 197, 64) | Fused attention kernel |

#### 2.4 Output Projection

| Step | Operation | Input Shape | Output Shape | Description |
|------|-----------|-------------|--------------|-------------|
| 1 | `transpose` | (N×12, 197, 64) | (197, N×12, 64) | Reorder for merge |
| 2 | `contiguous` | - | - | Memory contiguity |
| 3 | `view` | (197, N×12, 64) | (197×N, 768) | Merge heads |
| 4 | `linear` | (197×N, 768) | (197×N, 768) | out_proj: W×x + b |
| 5 | `view` | (197×N, 768) | (197, N, 768) | Reshape |

**Parameters:**
- out_proj.weight: (768, 768)
- out_proj.bias: (768,)

#### 2.5 Residual Connection

| Step | Operation | Input Shape | Output Shape | Description |
|------|-----------|-------------|--------------|-------------|
| 1 | `dropout` | (N, 197, 768) | (N, 197, 768) | Regularization |
| 2 | `add` | (N, 197, 768) | (N, 197, 768) | x + attention_output |

---

### 3. MLPBlock (Feed-Forward Network)

Two-layer MLP: 768 → 3072 → 768 with GELU activation.

| Step | Operation | Input Shape | Output Shape | Description |
|------|-----------|-------------|--------------|-------------|
| 1 | `linear` | (N, 197, 768) | (N, 197, 3072) | FC1: W₁×x + b₁ |
| 2 | `gelu` | (N, 197, 3072) | (N, 197, 3072) | GELU activation |
| 3 | `dropout` | (N, 197, 3072) | (N, 197, 3072) | Regularization |
| 4 | `linear` | (N, 197, 3072) | (N, 197, 768) | FC2: W₂×x + b₂ |
| 5 | `dropout` | (N, 197, 768) | (N, 197, 768) | Regularization |

**GELU Decomposition** (approximate):
```
gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
```

| Sub-step | Operation | Description |
|----------|-----------|-------------|
| 1 | `pow` | x³ |
| 2 | `mul` | 0.044715 × x³ |
| 3 | `add` | x + 0.044715×x³ |
| 4 | `mul` | √(2/π) × (...) |
| 5 | `tanh` | tanh(...) |
| 6 | `add` | 1 + tanh(...) |
| 7 | `mul` | x × (...) |
| 8 | `mul` | 0.5 × (...) |

**Parameters:**
- Linear 1 weight: (3072, 768)
- Linear 1 bias: (3072,)
- Linear 2 weight: (768, 3072)
- Linear 2 bias: (768,)

#### 3.1 Residual Connection

| Step | Operation | Input Shape | Output Shape | Description |
|------|-----------|-------------|--------------|-------------|
| 1 | `add` | (N, 197, 768) | (N, 197, 768) | x + mlp_output |

---

## Summary: All Low-Level Operations

### Compute Operations

| Category | Operations | Count per Block |
|----------|------------|-----------------|
| **Matrix Multiply** | `linear`, `bmm`, `baddbmm` | 5 (in_proj, out_proj, FC1, FC2, attention) |
| **Elementwise Binary** | `add`, `sub`, `mul`, `div` | ~20 |
| **Elementwise Unary** | `pow`, `rsqrt`, `tanh`, `gelu`, `softmax` | ~10 |
| **Reduction** | `mean` | 4 (LayerNorm) |

### Memory/Shape Operations

| Category | Operations |
|----------|------------|
| **Reshape** | `view`, `reshape`, `unflatten` |
| **Transpose** | `transpose`, `permute` |
| **Memory** | `contiguous`, `expand`, `cat` |
| **Indexing** | `squeeze`, `unsqueeze`, `index` |

### Complete Op List

```
add, baddbmm, bmm, cat, chunk, contiguous, div, dropout, expand,
gelu, index, linear (matmul + bias), mean, mul, permute, pow,
reshape, rsqrt, softmax, split, squeeze, sub, tanh, transpose,
unflatten, unsqueeze, view
```

---

## Parameter Count Summary

| Component | Parameters |
|-----------|------------|
| LayerNorm (×3 per block) | 2 × 768 × 3 = 4,608 |
| Self-Attention | (768×2304) + 2304 + (768×768) + 768 = 2,362,368 |
| MLP | (768×3072) + 3072 + (3072×768) + 768 = 4,722,432 |
| **Per Block Total** | ~7.1M |
| **Encoder Total (×12)** | ~85M |

---

## Source Files

- VisionTransformer: `torchvision/models/vision_transformer.py`
- MultiheadAttention: `torch/nn/modules/activation.py`
- multi_head_attention_forward: `torch/nn/functional.py`
- MLP: `torchvision/ops/misc.py`
