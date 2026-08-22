---
title: "Vulkan Backend Critical Rules"
category: BACKEND_CONSTRAINT
backends: [Vulkan]
last_validated: 2026-04-05
source_issues: [3922, 6373, 7343, 8288, 10602, 12634, 14507, 14984, 15441, 15700, 16823, 17299, 17366, 18696]
---

# Vulkan Backend — Critical Tribal Knowledge

1. **No runtime CPU fallback.** Once an op is lowered to Vulkan, it must execute on GPU. If a shader is missing, it crashes — not falls back. Always pair with XnnpackPartitioner for safety: `partitioner=[VulkanPartitioner(), XnnpackPartitioner()]`. [Source: #12634]

2. **PowerVR GPUs produce all-zero/NaN outputs with hardswish/hardsigmoid.** These ops get decomposed by `to_edge()` into primitives that fail on PowerVR. The native Vulkan shaders work, but decomposition bypasses them. Avoid `force_fp16` on PowerVR. [Source: #17299]

3. **Missing shader = crash, not warning.** `Could not find ShaderInfo` is fatal. Common missing shaders: `concat` with int types, `view_convert_buffer` with float/int32. Check if a PR fixes it before working around. [Source: #15441, #16823, #17366]

4. **Tensors must be rank <= 4.** Vulkan uses 3D textures; tensors with >4 dimensions cause `sizes_.size() <= 4 is false` crashes. Reshape before Vulkan delegation. [Source: #15441]

5. **Set `texture_limits` to match GPU's `maxImageDimension3D`.** The partitioner uses this to avoid delegating ops whose packed tensor extents exceed GPU limits. Default may not match your device. [Source: #17299]

6. **Debug Vulkan models with progressive slicing.** Export increasingly larger model slices (first N layers), run each on device, compare to XNNPACK baseline. This isolates the exact failing layer. [Source: #17299]

7. **Export from source, not pip, for Vulkan.** Pip-installed ExecuTorch may lack Vulkan custom ops (`et_vk.*`), causing `Missing operator` crashes. Build from source with `-DEXECUTORCH_BUILD_VULKAN=ON`. [Source: #17299]

8. **`VK_KHR_8bit_storage` is required for uint8 shaders.** Not all mobile GPUs support it. Models using int8 textures will crash on unsupported GPUs. Use FP16/FP32 as fallback. [Source: #16823]

9. **Ninja build fails on shader wildcards.** If you see `missing and no known rule to make it` for GLSL shaders, update to a version where the wildcard is expanded at configure time, or apply the fix from PR in #14984. [Source: #14984]

10. **Verify input preprocessing before blaming Vulkan.** Incorrect output may be caused by missing float conversion or normalization in app code, not the backend. Always `convertTo(CV_32F, 1.0/255.0)` for image models. [Source: #15490]

11. **`VulkanBackend is not registered` → use `--whole-archive`.** When linking `libvulkan_backend.a` into your binary, the linker drops static initialization. Use `-Wl,--whole-archive libvulkan_backend.a -Wl,--no-whole-archive`. [Source: #10602]

12. **SDPA + KV cache requires modifying op_registry.** For Llama models with `use_sdpa_with_kv_cache`, you need to add the relevant SDPA ops to `vulkan/op_registry.py`. [Source: #6373]

13. **NDK `glslc` may be insufficient.** Recent shader code uses `GL_EXT_integer_dot_product` which the NDK-bundled `glslc` may not support. Use the Vulkan SDK's `glslc` instead. [Source: #14507]

14. **Some models need `memory_layout_override`.** If a model (e.g., ResNet50) fails at runtime on Vulkan, try adding `"memory_layout_override": "channels_packed"` to `compile_options`. [Source: #3922]
