# WebGPU Backend — TODO

## Current State (Prototype)
- Single op: `aten.add.Tensor` (fp32, buffer storage)
- No Python AOT code — directly consumes Vulkan delegate (.pte exported via VulkanPartitioner)
- Reuses Vulkan FlatBuffer format (VH00 header + VK00 payload)
- Registers as `"VulkanBackend"` at runtime — mutually exclusive with Vulkan backend at link time
- Built-in WGSL shaders (not embedded in .pte)

## Architecture
```
VulkanPartitioner (Python) → VkGraphBuilder → VK00 FlatBuffer → .pte
    → WebGPU Runtime: registers as "VulkanBackend", parses VH00/VK00
    → WebGPUGraph::build → GPU buffers/pipelines/bind groups
    → WebGPUGraph::execute → encode + submit compute passes
```

Adding a new op requires only C++ runtime work:
1. WGSL shader + header
2. C++ op implementation (read args from VkGraph, create pipeline, record dispatch)
3. Register in CMakeLists.txt
4. Test with VulkanPartitioner export

## Performance: Command Encoding Overhead
WebGPU `GPUCommandBuffer` is single-use (no equivalent to Vulkan's cached command lists).
Per-dispatch API call cost adds up for large graphs.

**Primary mitigation: mega-kernel fusion.** Generate fused WGSL shaders for chains of
element-wise ops (add→relu→mul→clamp) at compile time. Embed via the existing
`shaders: [VkBytes]` field in schema.fbs.

## Next Steps
1. **More ops**: sub, mul, relu, linear (matmul), softmax, layer_norm
2. **fp16 support**: Feature-detect `shader-f16`, fallback to fp32
3. **Buffer pooling**: Reuse GPU buffers to avoid OOM at scale
4. **Pipeline caching**: Cache compiled pipelines across runs
5. **Profiling**: Wire WebGPU timestamp queries into ETDump/EventTracer
6. **LLM support**: KV cache management, Flash Attention in WGSL, quantized ops (int4/int8)
7. **Browser/JS runtime**: Emscripten build, JS harness, browser test page
