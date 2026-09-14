# Partitioner API

[VulkanPartitioner](https://github.com/pytorch/executorch/blob/main/backends/vulkan/partitioner/vulkan_partitioner.py)
is a Python class that controls what operators in a model can or should be
delegated to the Vulkan backend. It is the primary entrypoint to the Vulkan
backend and is also used to configure the behaviour of the Vulkan backend.

## Usage

For most use-cases, constructing `VulkanPartitioner()` with no arguments is
sufficient. In this case, the partitioner will lower as much of the model to
the Vulkan backend as possible.

```python
etvk_program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[VulkanPartitioner()],
).to_executorch()
```

## Common Config Options

Generally, the Vulkan backend is configured by passing a `compile_options`
dictionary to `VulkanPartitioner()`, i.e.

```python
compile_options = {
  "require_dynamic_shapes": True,
  "force_fp16": True,
}

etvk_program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[VulkanPartitioner(compile_options)],
).to_executorch()
```

### `require_dynamic_shapes`

If a model is expected to use dynamic shapes, then it is recommended to set the
`"required_dynamic_shapes"` key in `compile_options`.

Not all operators in Vulkan support dynamic shapes at the moment, although the
majority do. This flag will prevent operators that don't support dynamic shapes
from being lowered to Vulkan.

### `force_fp16`

This option causes the Vulkan backend to internally convert all FP32 tensors to
FP16. This can improve inference latency and memory footprint at the cost of
model accuracy.

FP32 input tensors will be automatically converted to FP16 upon entering the
Vulkan backend, and FP16 outputs will be automatically be converted to FP32 as
they are returned.
