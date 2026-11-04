# Op Support

The MLX backend supports ~90 ATen operators plus multi-node fused patterns and custom ops. The partitioner automatically determines which ops in your model can be delegated to MLX. Unsupported ops fall back to ExecuTorch's portable CPU runtime.

For the current list of supported operators and fused patterns, see the source:

- **[ops.py](https://github.com/pytorch/executorch/blob/main/backends/mlx/ops.py)** — Single-op handlers (ATen op → MLX IR node)
- **[patterns.py](https://github.com/pytorch/executorch/blob/main/backends/mlx/patterns.py)** — Multi-node fused patterns (quantized linear, SDPA, KV cache, etc.)

During lowering, the MLX partitioner prints a summary of supported and unsupported ops so you can see which ones are delegated and which fall back to CPU.

## Out-of-Range Indices in `gather` and `index_put`

Eager PyTorch raises an error when `gather` or `index_put` gets an index outside the tensor. The MLX backend does not, because a check would have to wait for the GPU in the middle of the graph. Instead, it clamps each index to `[-size, size - 1]` on the GPU, where `size` is the length of the indexed axis:

- A negative index counts from the end, so `-1` is the last position. Eager `index_put` does the same, but eager `gather` and `index_add_` reject negative indices.
- An index past either end reads or writes the first or last position.

For example:

```python
x = torch.arange(4.0).reshape(1, 4)
torch.gather(x, 1, torch.tensor([[9, -1, -7]]))
# eager: raises an error
# MLX:   reads positions 3, 3 and 0, so returns [[3., 3., 0.]]
```

So a bad index never reads or writes memory outside the tensor, but it is still a bug in the model:

- `gather` returns values from the wrong place.
- `index_put` overwrites real data at the first or last position, or adds to it with `accumulate=True`. For example, if an 8-slot cache is written as `cache[:, pos] = v` with `pos = torch.tensor([8])`, the write goes to slot 7.
- `index_add_` lowers to `index_put` with `accumulate=True`, so it adds to the first or last position.

Other ops that take indices behave differently, for example:

- `index_copy_` into a mutable buffer, and reading `x[i]` with a Python int, raise an error on an index outside `[-size, size - 1]`.
- `embedding`, `index_select`, `scatter_add`, and reading `x[idx]` with an index tensor do not check or clamp. An index outside `[-size, size - 1]` there can read or write memory outside the tensor.

To find bad indices, run the model in eager PyTorch on the CPU first, where they raise. Keep run-time indices in range too, for example by stopping generation before a cache position reaches the cache size.

On an empty axis there is nothing to clamp to: `gather` with a non-empty index raises, and `index_put` returns the input unchanged.
