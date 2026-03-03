# JetBlock Example

This directory contains an example implementation of **JetBlock**, a Gated Delta Rule based attention mechanism from NVIDIA's [Jet-Nemotron](https://github.com/NVIDIA/Jet-Nemotron) model.

## Overview

JetBlock implements efficient linear attention using the **Gated Delta Rule**, which maintains a recurrent state matrix that enables O(1) memory inference. This is particularly useful for long-context scenarios on edge devices.

### Key Features

- **Gated Delta Rule**: Efficient recurrent attention with linear complexity
- **Dynamic Short Convolution**: Position-dependent convolution kernels for local context mixing
- **No Quadratic Attention**: Avoids the O(n²) complexity of traditional attention
- **O(1) Memory Inference**: Constant memory usage when using KV cache during inference

## Architecture

The JetBlock consists of:

1. **Query/Key/Value Projections**: Standard linear projections
2. **Dynamic Convolution**: Generates position-dependent convolution kernels
3. **Gated Delta Rule Attention**: Recurrent state-based attention mechanism
4. **Gated RMSNorm**: Fused normalization with gating

### Gated Delta Rule

The core update equation is:
```
S_t = g_t * S_{t-1} + beta_t * k_t^T * (v_t - S_{t-1}^T * k_t)
o_t = S_t^T * q_t
```

Where:
- `S_t` is the recurrent state matrix
- `g_t` is the decay gate (controls how much of the previous state to retain)
- `beta_t` is the update strength
- `k_t`, `v_t`, `q_t` are the key, value, and query at timestep t

## Usage

### Basic Usage

```python
from executorch.examples.models.jet_nemotron import JetBlockModel

# Create model wrapper
model_wrapper = JetBlockModel(
    hidden_size=1536,
    num_layers=1,
    num_heads=6,
    head_dim=256,
)

# Get the eager PyTorch model
model = model_wrapper.get_eager_model()

# Get example inputs
inputs = model_wrapper.get_example_inputs()

# Run inference
output = model(*inputs)
print(output.shape)  # [batch, seq_len, hidden_size]
```

### Using JetBlock Directly

```python
from executorch.examples.models.jet_nemotron.jet_block import JetBlock, JetBlockConfig
import torch

# Configure JetBlock
config = JetBlockConfig(
    num_heads=6,
    head_dim=256,
    expand_v=2.0,
    conv_size=4,
)

# Create JetBlock layer
jet_block = JetBlock(
    hidden_size=1536,
    config=config,
    layer_idx=0,
)

# Forward pass
hidden_states = torch.randn(1, 128, 1536)
output, recurrent_state, conv_state = jet_block(hidden_states, use_cache=True)
```

### Export to ExecutorTorch

```python
import torch
from executorch.exir import to_edge

# Get model and inputs
model_wrapper = JetBlockModel()
model = model_wrapper.get_eager_model()
inputs = model_wrapper.get_example_inputs()

# Export with torch.export
exported = torch.export.export(model, inputs)

# Convert to Edge IR
edge_program = to_edge(exported)

# Lower to ExecutorTorch
et_program = edge_program.to_executorch()

# Save the .pte file
with open("jetblock.pte", "wb") as f:
    f.write(et_program.buffer)
```

## Configuration

The `JetBlockConfig` dataclass supports the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | str | "chunk" | Processing mode: "chunk" or "fused_recurrent" |
| `expand_v` | float | 2.0 | Value expansion factor |
| `num_heads` | int | 6 | Number of attention heads |
| `head_dim` | int | 256 | Dimension per head |
| `norm_eps` | float | 1e-5 | Epsilon for RMSNorm |
| `conv_size` | int | 4 | Kernel size for dynamic convolution |
| `dconv_generator_reduction` | int | 8 | Reduction factor for conv kernel generator |

## Files

- `jet_block.py`: Core JetBlock implementation with all dependencies
- `model.py`: ExecutorTorch-compatible model wrapper
- `__init__.py`: Package exports
- `BUCK`: Build configuration

## References

- [Jet-Nemotron GitHub](https://github.com/NVIDIA/Jet-Nemotron)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)
- Original paper: "Jet: A Modern Transformer-Based Normalizing Flow" (if applicable)

## License

This implementation is based on NVIDIA's Jet-Nemotron, which is licensed under Apache License 2.0.
