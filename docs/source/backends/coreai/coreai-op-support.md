# Op Support

The Core AI partitioner delegates every op that [coreai-torch](https://pypi.org/project/coreai-torch/) can convert, and leaves the rest to ExecuTorch's portable CPU kernels. It reads the list of supported ops directly from coreai-torch, so support always matches the coreai-torch version pinned in `backends/apple/coreai/requirements.txt`. It also keeps ExecuTorch from breaking up ops that Core AI runs as a whole, such as fused scaled dot-product attention.

For how the supported set is determined, see the partitioner's support check in [partitioner.py](https://github.com/pytorch/executorch/blob/main/backends/apple/coreai/partition/partitioner.py).

During lowering, the partitioner logs each op it leaves out, at `INFO` level:

```
Core AI cannot lower op, leaving out of delegate: <op>
```

## Delegate Boundary

Some nodes stay out of the delegate even when coreai-torch supports their op:

- **Python and symbolic scalars:** values can only pass into and out of a delegate as tensors, so nodes that take or produce a plain number run on the CPU.
- **64-bit tensors:** Core AI supports dtypes up to 32 bits. `get_default_passes()` casts `int64` and `float64` model inputs to 32 bits as they enter Core AI, so only that cast runs on the CPU and the model's own input dtypes don't change. Other ops that produce 64-bit tensors, such as `torch.arange` with its default `int64` result, stay on the CPU.
- **Nodes marked with `do_not_delegate`:** see [Excluding nodes from delegation](coreai-partitioner.md#excluding-nodes-from-delegation).

At runtime, delegate inputs and outputs must be `float16`, `float32`, `int32` or `bool` tensors that are contiguous, use the default dim order and are in CPU memory.
