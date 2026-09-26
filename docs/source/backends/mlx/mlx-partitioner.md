# Partitioner API

The MLX partitioner API allows for configuration of model delegation to the MLX backend. Passing an `MLXPartitioner` instance with no additional parameters will run as much of the model as possible on the MLX backend with default settings. This is the most common use case.

## Usage

```python
import torch
from executorch.backends.mlx import MLXPartitioner
from executorch.exir import to_edge_transform_and_lower

et_program = to_edge_transform_and_lower(
    torch.export.export(model, example_inputs),
    partitioner=[MLXPartitioner()],
).to_executorch()
```

::::{important}
`MLXPartitioner` must be used with `to_edge_transform_and_lower()`. The legacy `to_edge()` + `to_backend()` workflow is **not supported** because it decomposes ops that MLX has optimized implementations for.
::::

## Unsupported Op Logging

During partitioning, the partitioner logs a summary of any unsupported ops. This is useful for understanding what will fall back to CPU:

```
================================================================================
MLX Partitioner: UNSUPPORTED OPS SUMMARY
================================================================================
  [UNSUPPORTED x2] aten.some_op.default
      Reason: No handler registered
================================================================================
```

If all ops are supported, you'll see:

```
  (All call_function nodes are supported!)
```

Set `ET_MLX_DEBUG=1` to see detailed per-node support decisions during partitioning.
