# Op Support

The MLX backend supports ~90 ATen operators plus multi-node fused patterns and custom ops. The partitioner automatically determines which ops in your model can be delegated to MLX. Unsupported ops fall back to ExecuTorch's portable CPU runtime.

For the current list of supported operators and fused patterns, see the source:

- **[ops.py](https://github.com/pytorch/executorch/blob/main/backends/mlx/ops.py)** — Single-op handlers (ATen op → MLX IR node)
- **[patterns.py](https://github.com/pytorch/executorch/blob/main/backends/mlx/patterns.py)** — Multi-node fused patterns (quantized linear, SDPA, KV cache, etc.)

During lowering, the MLX partitioner prints a summary of supported and unsupported ops so you can see which ones are delegated and which fall back to CPU.

## Out-of-Range Indices in `gather` and `index_put`

Eager PyTorch raises an error when `gather` or `index_put` gets an index outside the tensor. The MLX backend does not, because a check would have to wait for the GPU in the middle of the graph. Instead, it clamps each index to `[-size, size - 1]` on the GPU, where `size` is the length of the indexed axis:

- A negative index counts from the end, so `-1` is the last position. Eager `index_put` does the same, but many other eager index ops, such as `gather`, `index_add_`, `index_copy_` and `embedding`, reject negative indices. On MLX, the other index ops below also count from the end.
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
- `index_add_` lowers to `index_put` with `accumulate=True`, so it treats a bad index the same way, but adds instead of overwriting.
- `index_copy` and `index_copy_` also lower to `index_put` and overwrite the first or last position. The exception is a method where every `index_copy` or `index_copy_` is the last write to a mutable buffer or input. There they raise an error on an index outside `[-size, size - 1]`.

Other ops that take indices behave differently, for example:

- Reading `x[i]` with a Python int raises an error on an index outside `[-size, size - 1]`.
- `embedding`, `index_select`, `scatter_add`, and reading `x[idx]` with an index tensor do not check or clamp. An index outside `[-size, size - 1]` there can read or write memory outside the tensor.

When the tensor that `index_put` or `scatter_add` writes has an 8-byte dtype, such as `int64` or `float64`, the MLX backend does not run it. It runs on the CPU instead. There `index_put` on a non-empty tensor raises an error on an index outside `[-size, size - 1]`, and `scatter_add` on an index outside `[0, size - 1]`. This also applies to `index_add_`, and to `index_copy` and `index_copy_` where they lower to `index_put`, so a negative index there still counts from the end.

To find bad indices, run the model in eager PyTorch on the CPU first, where they raise. Keep run-time indices in range too, for example by stopping generation before a cache position reaches the cache size.

On an empty axis there is nothing to clamp to: `gather` with a non-empty index raises, and `index_put` returns the input unchanged.
