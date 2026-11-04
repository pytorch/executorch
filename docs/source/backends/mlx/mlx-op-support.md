# Op Support

The MLX backend supports ~90 ATen operators plus multi-node fused patterns and custom ops. The partitioner automatically determines which ops in your model can be delegated to MLX. Unsupported ops fall back to ExecuTorch's portable CPU runtime.

For the current list of supported operators and fused patterns, see the source:

- **[ops.py](https://github.com/pytorch/executorch/blob/main/backends/mlx/ops.py)** — Single-op handlers (ATen op → MLX IR node)
- **[patterns.py](https://github.com/pytorch/executorch/blob/main/backends/mlx/patterns.py)** — Multi-node fused patterns (quantized linear, SDPA, KV cache, etc.)

During lowering, the MLX partitioner prints a summary of supported and unsupported ops so you can see which ones are delegated and which fall back to CPU.

## Out-of-Range Indices in `gather` and `index_put`

Eager PyTorch raises an error when `gather` or `index_put` gets an index outside the tensor. The MLX backend does not, because a check would have to wait for the GPU in the middle of the graph. Instead, it clamps each index to `[-size, size - 1]` on the GPU, where `size` is the length of the indexed axis:

- A negative index counts from the end, so `-1` is the last element. Eager `index_put` does the same, but eager `gather` rejects negative indices.
- An index past either end reads or writes the first or last element.

For example:

```python
x = torch.arange(4.0).reshape(1, 4)
torch.gather(x, 1, torch.tensor([[9, -1, -7]]))
# eager: raises an error
# MLX:   reads positions 3, 3 and 0, so returns [[3., 3., 0.]]
```

So a bad index gives a wrong result instead of an error, but it never reads or writes memory outside the tensor. If your model can produce bad indices, test it in eager PyTorch first.

On an empty axis there is nothing to clamp to: `gather` with a non-empty index raises, and `index_put` returns the input unchanged.
