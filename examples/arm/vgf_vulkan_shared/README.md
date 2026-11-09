# Shared VGF and Vulkan Backend Example

## Overview

This example demonstrates how the **VGF and Vulkan backends in ExecuTorch can execute different parts of a single model while sharing the same Vulkan device**.

The model is partitioned between the two GPU backends:

- **VGF backend** executes the first operation.
- **Vulkan backend** executes the intermediate operation.
- **VGF backend** executes the final operations.

Both backends use a shared `VkDevice`, avoiding the need to create separate Vulkan devices for each backend.

The example demonstrates:

- Executing a single ExecuTorch program with both VGF and Vulkan delegates.
- Creating and registering a Vulkan device with the features required by VGF.
- Reusing the registered device across both backends.
- Synchronizing access to a shared Vulkan queue.
- Verifying the numerical correctness of the mixed-backend execution.

The example does not implement zero-copy transfers between delegates or combine their execution into a single GPU submission. Ordinary ExecuTorch delegate boundaries remain.

## Example Model

The model is intentionally small to make backend partitioning and shared-device behavior easy to verify.

It performs the following operations:

```text
Input tensor [1, 3, 8, 8]
          |
          v
    +-------------+
    | avg_pool2d  |  VGF backend
    | kernel = 2  |
    | stride = 2  |
    +-------------+
          |
          v
    +-------------+
    |   sigmoid   |  Vulkan backend
    +-------------+
          |
          v
    +-------------+
    |  x * x      |  VGF backend
    |  + 0.25     |
    +-------------+
          |
          v
Output tensor [1, 3, 4, 4]
```

The exporter uses `VgfPartitioner` and `VulkanPartitioner` to assign operations to their respective backends.

For demonstration purposes, `sigmoid` is explicitly excluded from VGF partitioning so that Vulkan executes it. This is an example-specific partitioning policy, not a limitation of VGF's operator support.

The generated ExecuTorch execution plan must contain exactly three delegate calls:

```text
VgfBackend -> VulkanBackend -> VgfBackend
```

The exporter also verifies that no CPU `KernelCall` instructions remain in the execution plan.

## How Device Sharing Works

The example runner creates and registers a Vulkan device before loading the ExecuTorch model.

The device is created with the Vulkan extensions, features, and queue capabilities required by both backends.

The runner uses the following shared-device configuration:

```text
vulkan_shared_device_name = "vgf-vulkan-e2e"
vulkan_shared_group_id    = 7
vulkan_shared_device_mode = "lookup_only"
```

The device name and group ID identify the registry entry. Both backends use the same configuration to resolve the shared device.

The `lookup_only` mode requires the device to exist in the registry. If no compatible device is registered, initialization fails rather than silently creating another device.

The runner first verifies that the Vulkan adapter resolves the registered device. It then loads and executes the mixed-backend model.

Conceptually, the execution looks like this:

```text
                    Application
                         |
                         v
               Shared Vulkan Registry
                         |
                         v
                  Shared VkDevice
                         |
                Shared Vulkan Queue
                         |
           +-------------+-------------+
           |                           |
           v                           v
      VGF Backend                Vulkan Backend
           |                           |
           +-------------+-------------+
                         |
                         v
                 ExecuTorch Program
                         |
                         v
                VGF -> Vulkan -> VGF
```

### Queue Synchronization

Because both backends can submit commands to the same Vulkan queue, submissions must be synchronized.

VGF uses the mutex associated with the shared queue in the Vulkan registry. Queue ownership is established only while a synchronized operation is active.

The current implementation performs synchronous operations, including the required waits. Sharing the queue does not imply asynchronous execution or automatic overlap between delegates.

### Device Lifetime

The shared-device registry maintains references to registered devices.

Each VGF delegate owns its private resources, such as its command pool, but does not destroy a borrowed Vulkan device.

The example unregisters the device when its execution resources are released.

## Prerequisites

The example requires:

- An ExecuTorch checkout with VGF and Vulkan backend support.
- Python with the ExecuTorch export dependencies installed.
- Arm Model Converter and compatible VGF tools.
- Vulkan SDK and shader compiler tools.
- CMake and Ninja.
- A Vulkan implementation supporting the required Arm tensor and data-graph extensions, or a compatible ML emulation environment.

For native compilation, `VGF_ROOT` must reference a VGF installation containing:

```text
include/vgf/decoder.hpp
lib/libvgf.a
```

For Android cross-compilation, use a VGF library built for Android AArch64 rather than the macOS host library.

A combined VGF/Vulkan build requires Vulkan 1.3 support and a queue family supporting both `VK_QUEUE_COMPUTE_BIT` and `VK_QUEUE_DATA_GRAPH_BIT_ARM`.

## Export the Model

From the ExecuTorch repository root, run:

```bash
python -m examples.arm.vgf_vulkan_shared.export \
  --output-dir ./vgf_vulkan_shared_out
```

If required by your Model Converter installation, additional compiler options can be supplied through repeated `--compiler-flag=...` arguments.

The exporter writes:

```text
vgf_vulkan_shared_out/
├── vgf_vulkan_vgf.pte
├── input0.bin
├── expected0.bin
├── edge_graph.txt
└── manifest.json
```

These files contain:

| File | Description |
|---|---|
| `vgf_vulkan_vgf.pte` | ExecuTorch program containing VGF and Vulkan delegates |
| `input0.bin` | FP32 input tensor |
| `expected0.bin` | Reference output calculated using PyTorch |
| `edge_graph.txt` | Lowered graph for inspecting backend partitioning |
| `manifest.json` | Export metadata |

Input and reference-output tensors use contiguous, little-endian FP32 storage.

Before writing the `.pte` file, the exporter validates the execution plan and checks that the expected VGF → Vulkan → VGF sequence was generated.

## Build and Run

Choose **one** of the two build options below. Native and Android builds are alternatives for the same `vgf_vulkan_shared_runner` executable; neither is a prerequisite for the other. Both options use the model exported above.

Initialize the Vulkan dependencies for either build option:

```bash
git submodule update --init \
  backends/vulkan/third-party/Vulkan-Headers \
  backends/vulkan/third-party/volk \
  backends/vulkan/third-party/VulkanMemoryAllocator
```

### Option A: Native Build and Run

Configure the build:

```bash
export VGF_ROOT=/path/to/vgf/install
export BUILD=cmake-out-vgf-vulkan

cmake -S . -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DEXECUTORCH_BUILD_VGF=ON \
  -DEXECUTORCH_BUILD_VULKAN=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_VGF_SHARED_EXAMPLE=ON \
  -DEXECUTORCH_ENABLE_LOGGING=ON \
  -DEXECUTORCH_VGF_ZERO_COPY_IO=OFF \
  -DEXECUTORCH_VGF_ROOT="$VGF_ROOT"
```

Build the example runner:

```bash
cmake --build "$BUILD" \
  --target vgf_vulkan_shared_runner \
  --parallel 8
```

Locate the executable:

```bash
RUNNER="$(find "$BUILD" -type f \
  -name vgf_vulkan_shared_runner -print -quit)"
```

Run the model:

```bash
"$RUNNER" ./vgf_vulkan_shared_out
```

On systems without native Arm tensor and data-graph support, configure the Vulkan ML emulation layers before launching the runner.

### Option B: Android Build and Deployment

#### Cross-compile for Android

To execute the example on an Android device with Arm tensor and data-graph support, cross-compile the runner using the Android NDK.

Set the Android NDK and target VGF installation:

```bash
export ANDROID_NDK=/path/to/android-ndk
export VGF_ROOT=/path/to/android/vgf/install
export BUILD=cmake-out-android-vgf-vulkan
```

Configure CMake:

```bash
cmake -S . -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DANDROID_STL=c++_static \
  -DBUILD_SHARED_LIBS=OFF \
  -DEXECUTORCH_BUILD_VGF=ON \
  -DEXECUTORCH_BUILD_VULKAN=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_VGF_SHARED_EXAMPLE=ON \
  -DEXECUTORCH_ENABLE_LOGGING=ON \
  -DEXECUTORCH_VGF_ZERO_COPY_IO=OFF \
  -DEXECUTORCH_VGF_ROOT="$VGF_ROOT"
```

Build the runner:

```bash
cmake --build "$BUILD" \
  --target vgf_vulkan_shared_runner \
  --parallel 8
```

Find the executable and shared runtime library:

```bash
RUNNER="$(find "$BUILD" -type f \
  -name vgf_vulkan_shared_runner -print -quit)"

REGISTRY="$(find "$BUILD" -type f \
  -name libexecutorch_vulkan_shared_runtime.so -print -quit)"

test -n "$RUNNER" && test -n "$REGISTRY"
```

The runner must be compiled for Android AArch64. All participating backends must use the same shared Vulkan registry library.

#### Deploy and Run on Android

Connect the device using ADB:

```bash
adb devices
```

Create a directory on the device:

```bash
adb shell mkdir -p /data/local/tmp/vgf-vulkan-shared
```

Deploy the runner and the registry library:

```bash
adb push "$RUNNER" "$REGISTRY" \
  /data/local/tmp/vgf-vulkan-shared/
```

Deploy the exported model and test tensors:

```bash
adb push ./vgf_vulkan_shared_out/. \
  /data/local/tmp/vgf-vulkan-shared/
```

Run the example:

```bash
adb shell '
  cd /data/local/tmp/vgf-vulkan-shared &&
  chmod 755 vgf_vulkan_shared_runner &&
  LD_LIBRARY_PATH=/data/local/tmp/vgf-vulkan-shared \
    ./vgf_vulkan_shared_runner .
'
```

If the runner has additional shared-library dependencies, deploy them to the same directory or make them available through the Android linker configuration.

## Verify the Results

The runner checks that the Vulkan adapter resolves the same `VkDevice` registered for the VGF backend.

Expected output includes messages similar to:

```text
Shared VkDevice verified:
  VGF=<device handle>
  Vulkan=<same device handle>

PASS iteration=0 max_abs_error=...
PASS iteration=1 max_abs_error=...
PASS iteration=2 max_abs_error=...
```

The actual device handles and numerical errors depend on the runtime environment.

The runner executes the model three times and compares the output with the PyTorch reference using:

```text
atol = 1e-5
rtol = 1e-4
```

It also saves the first output to:

```text
actual0.bin
```

To retrieve this output from Android:

```bash
adb pull \
  /data/local/tmp/vgf-vulkan-shared/actual0.bin \
  ./vgf_vulkan_shared_out/actual0.bin
```

Successful execution demonstrates that:

1. Both backends can operate with the registered Vulkan device.
2. The mixed VGF → Vulkan → VGF execution plan runs without CPU operator fallback.
3. The output matches the PyTorch reference within the specified tolerance.

## Troubleshooting

### Model export fails

Verify that the Python environment contains the required ExecuTorch dependencies and compatible Arm Model Converter tools.

If the exporter reports unexpected backend partitioning, inspect `edge_graph.txt` and confirm that the expected delegate sequence is present.

### Shared device cannot be resolved

Verify that the device is registered before delegate initialization and that both backends use matching registry keys.

Also check that the registered device has the required Vulkan extensions, enabled feature bits, and queue capabilities.

### Vulkan extensions are unavailable

The native GPU must support the Arm tensor and data-graph extensions required by VGF.

On a development machine without native support, use a compatible Vulkan ML emulation environment.

### Runner cannot load a shared library

Check that `libexecutorch_vulkan_shared_runtime.so` is deployed and accessible through `LD_LIBRARY_PATH`.

Ensure that all backends use the same registry library and are built from compatible runtime sources.

### Numerical output differs

Confirm that the deployed `.pte`, input tensor, and reference output were produced by the same export.

Also verify that the selected GPU driver and Model Converter configuration support the operations used by the model.

## Summary

This example demonstrates mixed VGF and Vulkan execution using a single shared Vulkan device.

It provides a small, reproducible model with a known execution sequence:

```text
VGF -> Vulkan -> VGF
```

The example verifies device sharing, queue synchronization, backend interoperability, and numerical correctness.

It is intended as a starting point for applications that use VGF and Vulkan together in one ExecuTorch program while avoiding separate Vulkan device creation for each backend.
