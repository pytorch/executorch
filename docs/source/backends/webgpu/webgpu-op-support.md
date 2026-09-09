# Operator Support

The current source of truth for WebGPU operator support is the runtime registry under
[`backends/webgpu/runtime/ops`](https://github.com/pytorch/executorch/tree/main/backends/webgpu/runtime/ops).
Operator tests are under
[`backends/webgpu/test/ops`](https://github.com/pytorch/executorch/tree/main/backends/webgpu/test/ops),
and the generated native test catalog is defined in
[`backends/webgpu/test/op_tests/cases.py`](https://github.com/pytorch/executorch/blob/main/backends/webgpu/test/op_tests/cases.py).

## Compatibility Considerations

- `WebGPUPartitioner` is a thin wrapper over `VulkanPartitioner`; it preserves
  Vulkan partitioning behavior and does not add WebGPU capability policy.
- Support can depend on tensor rank, shape, layout, and data type in addition
  to the operator name.
- FP16 shader paths require a WebGPU adapter exposing `shader-f16`.
- Dynamic shapes allocate buffers at the exported maximum shape. Operators
  must also implement runtime resize propagation.
- Quantized and LLM-specific operators generally require the corresponding
  Vulkan custom-op export pattern.

Use the native operator-test framework to generate fixtures for specific
operators. For example, generate the verified `add` cases with:

```bash
python -m executorch.backends.webgpu.test.op_tests.generate_op_tests \
    --output /tmp/webgpu_op_tests --ops add
```

The full build-and-run flow is available through:

```bash
bash backends/webgpu/test/test_build_webgpu.sh
```
