---
title: "Vulkan Backend Known Issues and Workarounds"
category: DEBUGGING
backends: [Vulkan]
last_validated: 2026-04-05
source_issues: [3922, 6373, 7343, 8078, 10602, 11754, 12232, 12634, 14507, 14984, 15296, 15344, 15441, 15490, 15700, 16354, 16647, 16823, 17293, 17299, 17366, 17855, 18696]
---

# Vulkan Backend Known Issues and Workarounds

## GPU-Specific Issues

### PowerVR: All-Zero Outputs

**Symptom**: Models produce all-zero outputs on PowerVR D-Series GPUs (e.g., Google Pixel 10 Pro), while the same models work on Adreno GPUs and macOS/MoltenVK. [Source: #17299]

**Root cause**: `aten.hardswish` and `aten.hardsigmoid` are **decomposed into primitive ops** (`mul/add/clamp/div` with constant tensors) by PyTorch's default decomposition table during `to_edge()`. The Vulkan backend has native GLSL shaders for both ops that work correctly on PowerVR, but they never get invoked because the ops are already decomposed. The decomposed primitives produce NaN on PowerVR, which propagates as zeros. [Source: #17299]

**Debugging methodology — Progressive Model Slicing**:

The reporter used a systematic approach to isolate the failure: [Source: #17299]
1. Export increasingly larger slices of the model (first N layers)
2. Run each slice on the target device
3. Compare outputs to XNNPACK baseline
4. Identify the exact layer where outputs diverge

```python
# Example: Export first N layers of MobileNetV3-Small
for n in range(1, num_layers + 1):
    model_slice = ModelSlice(model, n)
    et = to_edge_transform_and_lower(
        torch.export.export(model_slice, inputs),
        partitioner=[VulkanPartitioner(compile_options={
            "texture_limits": (2048, 2048, 2048),
        })]
    )
    # Save and run on device, compare to XNNPACK
```

**Minimal single-op tests**: Export 13 minimal single-operator models to isolate exactly which operations produce incorrect results on the target GPU. [Source: #17299]

**Workaround**: Avoid `force_fp16: True` on PowerVR GPUs. Test with FP32 first to isolate precision issues from shader bugs.

### Missing VK_KHR_8bit_storage on Mobile GPUs

**Symptom**: `Shader image_to_nchw_texture3d_uint8_uint8 not compatible with device. Missing support for extension or physical device feature: VK_KHR_8bit_storage` [Source: #16823]

**Cause**: Some mobile GPUs lack the `VK_KHR_8bit_storage` Vulkan extension required for uint8 texture operations.

**Workaround**: Avoid uint8/int8 tensor types in models targeted at these GPUs. Use FP16 or FP32 instead.

### Adreno: Memory Allocation Failures

**Symptom**: `vmaCreateBuffer` fails with `VK_ERROR_OUT_OF_DEVICE_MEMORY` on Adreno GPUs (e.g., Adreno 650 on OnePlus 8 Pro) with larger input resolutions. [Source: #17366]

**Cause**: Model + intermediate tensors exceed GPU memory. The Vulkan backend may not yet optimize memory reuse efficiently.

**Workaround**: Reduce input resolution or try a smaller model variant.

### Adreno: Texture Tensor UBO Overflow

**Symptom**: `Vulkan uniform data allocation has exceeded tensor uniform buffer size` at model load time. [Source: #17293]

**Cause**: Tensor metadata exceeds the uniform buffer size limit.

**Fix**: Fixed by PR #17294 which increases the UBO size limit.

### Android 16 Crash: vkCreateComputePipelines Returns -3

**Symptom**: `vkCreateComputePipelines` returns -3 and crashes on Android 16 (API 36) system images in Android Studio emulator. Works on Android 14/15. [Source: #11754]

**Status**: Under investigation. Physical devices (Pixel 6) may not be affected. The issue appears to be emulator-specific on Android 16.

### Non-Deterministic Outputs (VAE Models)

**Symptom**: Model produces different outputs across runs with the same input, even without quantization. [Source: #15344]

**Cause**: No implicit FP16 conversion — non-determinism likely indicates a shader accessing out-of-bounds memory.

**Workaround**: Report the specific model and GPU combination for investigation.

## Shader Issues

### Missing Shader Variants

**Symptom**: `Could not find ShaderInfo with name <shader_name>` at runtime. [Source: #15441, #16823, #17366]

Common missing shaders:
- `concat_1_texture3d_int32` — concat with integer inputs [Source: #12634]
- `concat_3_texture3d_uint8` — concat with uint8 inputs [Source: #16823]
- `view_convert_buffer_float_int32` — view/reshape with float-to-int32 conversion [Source: #17366]

**Root cause**: The Vulkan partitioner lowers ops to the backend even when the required shader variant doesn't exist for the specific dtype combination.

**Fix**: These are addressed individually via PRs adding missing shader combinations (e.g., PR #17382 for view_convert_buffer). [Source: #17366]

**Workaround**: Use XNNPACK backend instead of Vulkan for models that trigger missing shaders.

### Tensor Rank > 4 Not Supported

**Symptom**: `(sizes_.size() <= 4) is false!` crash when running models with tensors that have more than 4 dimensions. [Source: #15441]

**Cause**: The Vulkan backend uses 3D textures for tensor storage and currently only supports up to 4D tensors.

**Workaround**: Restructure model to avoid 5D+ tensors in Vulkan-delegated subgraphs, or let those ops fall back to XNNPACK.

## Partitioner / Lowering Issues

### No Runtime CPU Fallback

Once an op is lowered to Vulkan by the partitioner, it **must** execute on the GPU. There is no runtime fallback to CPU for individual ops within a Vulkan delegate. [Source: #12634]

**Mitigation**: The Vulkan partitioner is being improved to reject ops with unsupported input/output dtypes at partition time. In the meantime, use dual partitioners:

```python
partitioner=[VulkanPartitioner(), XnnpackPartitioner()]
```

This ensures XNNPACK handles ops that Vulkan's partitioner rejects.

### Ops Lowered with Unsupported Dtypes

**Symptom**: Ops like `concat` with integer inputs are lowered to Vulkan despite lacking shader support. [Source: #12634]

**Status**: The partitioner's dtype validation is being strengthened to prevent this.

## Build Issues

### Ninja Wildcard DEPENDS Bug

**Symptom**: `ninja: error: '_deps/executorch/backends/vulkan/runtime/graph/ops/glsl/*', needed by 'vulkan_compute_shaders/spv.cpp', missing and no known rule to make it` [Source: #14984]

**Cause**: Ninja does not handle wildcards in `DEPENDS` directives in `ShaderLibrary.cmake`.

**Fix**: Fixed in recent versions. The shader glob is now expanded at CMake configure time rather than using wildcards.

### Missing glslc Compiler / NDK glslc Insufficient

**Symptom**: Vulkan shaders fail to compile during build, or `glslc: error: invalid value` for `GL_EXT_integer_dot_product`.

**Fix**: The `glslc` distributed with Android NDK may no longer be sufficient — it may not support required extensions. Install `glslc` from the Vulkan SDK and ensure it's on PATH. [Source: #14984, #14507]

### cstdint Include on GCC

**Symptom**: Build fails on Jetson or aarch64 Linux with `uint32_t was not declared in this scope` in `Types.h`. [Source: #7343]

**Fix**: Add `#include <cstdint>` to `backends/vulkan/runtime/graph/containers/Types.h`.

### VulkanMemoryAllocator Submodule Clone Failure

**Symptom**: `fatal: clone of 'https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git' failed` [Source: #3949]

**Cause**: Network issues (common behind firewalls/proxies).

**Fix**: Manually clone the submodule or configure git proxy settings.

### VMA Assertion Crash on macOS/MoltenVK

**Symptom**: VMA assertion crash on macOS when using MoltenVK: `test_host_cached_available()` returns `bool` but function declared as returning `VmaAllocationCreateFlags`. [Source: #18696]

**Cause**: Bug introduced in PR #17856 (Raspberry Pi 5 fix), cherry-picked into v1.2.0. The function's return type was incorrectly set to `bool`.

**Fix**: Fixed on `main` by PRs #18105 and #18726. Not yet in a release — may need cherry-pick into v1.2.1.

### Raspberry Pi 5: Linear Image Sampling Error

**Symptom**: `MESA: error: Sampling from linear image is not supported` on Raspberry Pi 5. [Source: #17855]

**Fix**: Addressed in PR #17856 — requires adjustments to image sampling in the Vulkan backend to work with the RPi 5's VideoCore VII GPU.

## Registration and Linking Issues

### VulkanBackend is Not Registered

**Symptom**: `Backend VulkanBackend is not registered` at runtime, despite linking `libvulkan_backend.a`.

**Cause**: Static initialization symbols are dropped by the linker. [Source: #10602]

**Fix**: Use the `--whole-archive` linker option:
```cmake
target_link_libraries(my_app
    -Wl,--whole-archive ${PATH_TO}/libvulkan_backend.a -Wl,--no-whole-archive
)
```

The Vulkan backend registers with ExecuTorch via static initialization — the linker will discard the registration code unless forced to include it.

### Module vs executor_runner: mlock Failure

**Symptom**: `mlock failed: Out of memory` when loading a model via `Module`, but the same model works with `vulkan_executor_runner`. [Source: #10602]

**Cause**: The `Module` class uses `MmapDataLoader` which calls `mlock` on the entire file, while `executor_runner` uses a different data loading strategy. Edge devices with limited memory may fail `mlock`.

**Workaround**: Use `Module::LoadMode::Mmap` or a custom data loader that doesn't `mlock` the entire file.

### Vulkan mean Errors During Lowering

**Symptom**: `torch.mean` errors out during `to_edge_transform_and_lower` with Vulkan partitioner. [Source: #12232]

**Workaround**: Use `torch.mean(x, dim=0, keepdim=True).squeeze(0)` instead of `torch.mean(x, dim=0)`.

## Model-Specific Issues

### YOLO-NAS Non-Deterministic Outputs

**Symptom**: YOLO-NAS model produces divergent outputs across runs with the same input on Vulkan. [Source: #15700]

**Root cause**: Bug in `split_with_sizes` operator. Fixed in PR #15793.

**Note**: After the fix, `executor_runner` may still show divergence due to a separate issue. Use the Python export/test flow for validation.

### U2Net Corrupted Output

**Symptom**: U2Net produces corrupted images on Vulkan while working on XNNPACK and CoreML. [Source: #15490]

**Investigation**: In some cases, the issue was in the **application code** (input not converted to float and normalized to [0,1] range), not in the Vulkan backend itself.

**Debugging step**: Always verify input preprocessing matches what the model expects:

```cpp
cv::Mat image;
image_orig.convertTo(image, CV_32F, 1.0 / 255.0);  // Convert to float, normalize
```

## Debugging Methodology

When debugging Vulkan backend issues: [Source: #17299, #15700]

1. **Verify with XNNPACK first**: If the model works with XNNPACK, the issue is Vulkan-specific
2. **Progressive model slicing**: Export increasingly larger model slices to find the failing layer
3. **Minimal single-op tests**: Export individual operators to isolate shader bugs
4. **Check FP32 before FP16**: Disable `force_fp16` to isolate precision from correctness issues
5. **Export from source, not pip**: Use source-built ExecuTorch for export to ensure all Vulkan custom ops are registered [Source: #17299]
6. **Check GPU capabilities**: Query `maxImageDimension3D`, supported extensions (`VK_KHR_8bit_storage`), and device memory limits
7. **Check SDPA op support**: If using SDPA + KV cache (e.g., Llama), verify that the relevant ops are in `vulkan/op_registry.py` [Source: #6373]
8. **Use `--clean` flag for environment issues**: If you get errors after updates, try `./install_executorch.sh --clean` instead of creating a new venv [Source: #14806]
