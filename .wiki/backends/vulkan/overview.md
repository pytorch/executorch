---
title: "Vulkan Backend Overview"
category: BACKEND_CONSTRAINT
backends: [Vulkan]
last_validated: 2026-04-05
source_issues: [1440, 3922, 6373, 7132, 7343, 8078, 8214, 8288, 10494, 10602, 11754, 11780, 12634, 12799, 12920, 14507, 14984, 15344, 15441, 15490, 15670, 15700, 16124, 16365, 16823, 17299, 17366, 17855, 18696]
---

# Vulkan Backend Overview

## What Is the Vulkan Backend

The Vulkan backend is ExecuTorch's **GPU delegate** that leverages the Vulkan graphics API for accelerated inference on mobile and edge GPUs. It uses GLSL compute shaders compiled to SPIR-V to execute neural network operators on the GPU. [Source: #8288]

The backend was designed primarily for **mobile GPU acceleration** (Adreno, Mali, PowerVR) but also works on desktop GPUs via standard Vulkan drivers and on macOS via MoltenVK. [Source: #17299, #8288]

## Supported Platforms and GPUs

| Platform | GPU Family | Status | Notes |
|----------|-----------|--------|-------|
| Android | Qualcomm Adreno | Primary target | CI-tested, best supported |
| Android | ARM Mali | Supported | May need `VK_KHR_8bit_storage` check [Source: #16823] |
| Android | PowerVR | Experimental | Known issues with all-zero outputs, hardswish/hardsigmoid decomposition [Source: #17299] |
| Linux | NVIDIA/AMD/Intel | Works | Not performance-optimized for server GPUs; shaders tuned for mobile [Source: #8288] |
| macOS | MoltenVK | Works | Useful for development/testing; VMA assertion crash bug on v1.2.0 fixed on main [Source: #17299, #18696] |
| Windows | Any Vulkan GPU | Untested in CI | Not usable from WSL; DirectML support requested but not planned [Source: #8078, #17298] |
| NVIDIA Jetson | Jetson Orin/Xavier | Works | May need `cstdint` include fix [Source: #7343] |
| Raspberry Pi 5 | VideoCore VII | Works | Needs fix for `Sampling from linear image is not supported` error [Source: #17855] |
| Google Pixel 9/10 | Tensor G4/Tensor G5 | Works with Vulkan | QNN backend not supported (not Qualcomm SoC); Vulkan is the recommended GPU backend [Source: #15670] |

## Architecture

1. **Export time**: The `VulkanPartitioner` identifies ops that can run on Vulkan and creates delegate blobs
2. **Shader compilation**: GLSL shaders are compiled to SPIR-V at build time using `glslc`
3. **Runtime**: The Vulkan backend loads compute shaders, creates GPU buffers/textures, and dispatches workgroups

The backend uses 3D textures for tensor storage with texel packing along the channel dimension. This means `maxImageDimension3D` of the GPU limits tensor sizes. [Source: #17299]

## Export Flow

```python
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner

et_program = to_edge_transform_and_lower(
    exported,
    partitioner=[VulkanPartitioner()]
)
```

### With Options

```python
VulkanPartitioner(compile_options={
    "texture_limits": (2048, 2048, 2048),  # Match GPU's maxImageDimension3D
    "force_fp16": True,                     # Use half precision (caution on PowerVR)
    "memory_layout_override": "channels_packed",  # May be required for some models [Source: #3922]
})
```

**Note on `memory_layout_override`**: Some models (e.g., ResNet50) require explicitly setting this option to avoid runtime errors. [Source: #3922]

### With CPU Fallback

```python
# Vulkan first, XNNPACK handles unsupported ops
partitioner=[VulkanPartitioner(), XnnpackPartitioner()]
```

**Note**: There is no runtime CPU fallback — ops that the partitioner lowers to Vulkan must execute on GPU. If an op is lowered but unsupported at runtime, it will crash. [Source: #12634]

## Building

### Desktop (Linux/macOS)

```bash
CMAKE_ARGS="-DEXECUTORCH_BUILD_VULKAN=ON" ./install_executorch.sh -e
```

Or manually:

```bash
cmake -DEXECUTORCH_BUILD_VULKAN=ON \
      -DCMAKE_BUILD_TYPE=Release \
      ...
```

Requires `glslc` (from Vulkan SDK or Android NDK) to be on PATH. [Source: #14984]

### Android Cross-Compilation

```bash
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a \
      -DEXECUTORCH_BUILD_VULKAN=ON \
      ...
```

**Known build issue**: Ninja cannot handle wildcards in shader `DEPENDS` paths. Fixed in recent versions. [Source: #14984]

**glslc version requirement**: As of recent versions, the `glslc` bundled with Android NDK may not be sufficient — it may not support `GL_EXT_integer_dot_product` extension required for some shaders. Install `glslc` from the Vulkan SDK instead. [Source: #14507]

### Jetson (aarch64)

May need to add `#include <cstdint>` to `backends/vulkan/runtime/graph/containers/Types.h` on older GCC versions. [Source: #7343]

## Integration

The Vulkan backend registers with ExecuTorch via static initialization. No extra runtime setup is needed — just link the Vulkan backend library and load a Vulkan-delegated PTE. [Source: #10602]

```cmake
# In your CMakeLists.txt
set(EXECUTORCH_BUILD_VULKAN ON)
# Link against vulkan_backend target
```

## Quantization Support

The Vulkan backend supports quantized models (since v0.5): [Source: #7132]

- **4-bit quantization**: Supported for weight-only quantization (e.g., Llama models)
- **8-bit quantization**: Supported for int8 tensor operations (requires `VK_KHR_8bit_storage`)

Use the standard PT2E quantization flow — there is no Vulkan-specific quantizer. Quantization docs for Vulkan are being updated.

## Performance Characteristics

- Shaders are currently **optimized for mobile GPUs** (Adreno, Mali); desktop GPU performance will be suboptimal [Source: #8288]
- Focus areas for optimization: 4-bit weight quantized matmul for Transformer models [Source: #8288]
- Memory-bound workloads benefit most from GPU execution; small models may not see speedup due to dispatch overhead
- **LLM performance on Samsung Galaxy S24**: ~260 tok/s prefill, ~34 tok/s decode with quantized Llama 3.2 1B [Source: #12920]
- **Slow `to_dim_order_copy.out` for FP16**: This op is called frequently (2x more for FP16 than FP32 in MobileNetV3) and is disproportionately slow for FP16 [Source: #12799]

## SDPA and KV Cache

To use SDPA with KV cache on Vulkan (e.g., for Llama), you need to modify the op registry in `vulkan/op_registry.py` to include the relevant SDPA ops. [Source: #6373]

## Weight Sharing

Weight sharing across multiple entry points is **not yet supported** in the Vulkan backend (unlike XNNPACK which has `ENABLE_XNNPACK_WEIGHTS_CACHE`). [Source: #11780]

## Current Focus and Roadmap

- **Operator coverage expansion**: Adding missing shader variants (e.g., `view_convert_buffer` float/int32) [Source: #17366]
- **Improved partitioner validation**: Preventing unsupported ops from being lowered to Vulkan [Source: #12634, #16823]
- **PowerVR GPU support**: Investigating decomposition-related NaN issues [Source: #17299]
- **Integer tensor support**: Adding int32/int64 support for ops like `concat` [Source: #12634]
- **Shared library (`.so`) support**: Making vulkan_backend loadable at runtime instead of compiled into `libexecutorch_jni.so` [Source: #10494]
- **`to_dim_order_copy` Vulkan impl**: To avoid graph breaks from this op falling back to CPU [Source: #12921]
