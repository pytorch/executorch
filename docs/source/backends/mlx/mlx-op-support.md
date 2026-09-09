# Op Support

The MLX backend supports ~90 ATen operators plus multi-node fused patterns and custom ops. The partitioner automatically determines which ops in your model can be delegated to MLX. Unsupported ops fall back to ExecuTorch's portable CPU runtime.

For the current list of supported operators and fused patterns, see the source:

- **[ops.py](https://github.com/pytorch/executorch/blob/main/backends/mlx/ops.py)** — Single-op handlers (ATen op → MLX IR node)
- **[patterns.py](https://github.com/pytorch/executorch/blob/main/backends/mlx/patterns.py)** — Multi-node fused patterns (quantized linear, SDPA, KV cache, etc.)

During lowering, the MLX partitioner prints a summary of supported and unsupported ops so you can see which ones are delegated and which fall back to CPU.
