# Shared VGF/Vulkan example

Export and execute `VGF -> Vulkan -> VGF` in one ExecuTorch program.
The delegates share a VGF-capable `VkDevice` and queue; command pools and
per-delegate resources remain independent. This example does **not** implement
GPU-to-GPU zero-copy or fused submissions.

## Requirements and registration contract

Use the ExecuTorch Python environment, Vulkan SDK, Model Converter and ML SDK
matching this checkout. Source `examples/arm/arm-scratch/setup_path.sh` on the
host after the normal Arm setup. Native execution requires a VGF-capable driver
or the Graph/Tensor emulation layers; Android needs a compatible driver/device.
VGF requires Vulkan 1.3. Shared execution also requires a queue supporting both
compute and data graph operations.

Before loading a shared method, call `resolve_vgf_shared_device` with
`device_mode = SharedDeviceMode::kCreateOnly`, then pass the same device name and
group to both delegates with `vulkan_shared_device_mode = "lookup_only"`.
Keep the registration and device owner alive until all consumers are destroyed.
The runner demonstrates this contract and checks shared device identity.
An arbitrary Vulkan-only device cannot be upgraded after creation to enable VGF.
The canonical instance version is negotiated independently of backend order;
the existing eager Vulkan default adapter is retained.

## Export

```bash
source examples/arm/arm-scratch/setup_path.sh
python -m examples.arm.vgf_vulkan_shared.export --output-dir vgf_vulkan_shared_out
```

The output contains `vgf_vulkan_vgf.pte`, little-endian FP32 `input0.bin` and
`expected0.bin`, `edge_graph.txt`, and `manifest.json`. Input shape is
`[1, 3, 8, 8]`; output shape is `[1, 3, 4, 4]`. Export asserts the delegate order
and rejects CPU `KernelCall` fallback.

## Build

From the repository root, choose a native or Android build directory. Both
alternatives use the same exported files.

```bash
BUILD=cmake-out/vgf-vulkan-shared
TOOLCHAIN_ARGS=()
# For Android instead, set ANDROID_NDK and uncomment:
# BUILD=cmake-out/vgf-vulkan-shared-android
# TOOLCHAIN_ARGS=(-DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake"
#                 -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28)

cmake -S . -B "$BUILD" "${TOOLCHAIN_ARGS[@]}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_VGF=ON \
  -DEXECUTORCH_BUILD_VULKAN=ON \
  -DEXECUTORCH_BUILD_VGF_SHARED_EXAMPLE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DPYTHON_EXECUTABLE="$(command -v python3)" \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build "$BUILD" --target vgf_vulkan_shared_runner --parallel
RUNNER="$(find "$BUILD" -type f -name vgf_vulkan_shared_runner -print -quit)"
```

## Run

For a native build, with the host driver/emulation environment configured:

```bash
"$RUNNER" vgf_vulkan_shared_out
```

For an Android build, deploy the executable, exported files, the matching
`libexecutorch_vulkan_shared_runtime.so`, and the NDK's arm64 `libc++_shared.so`:

```bash
DEVICE_DIR=/data/local/tmp/vgf-vulkan-shared
adb shell mkdir -p "$DEVICE_DIR"
adb push "$RUNNER" "$DEVICE_DIR/"
adb push vgf_vulkan_shared_out/. "$DEVICE_DIR/"
SHARED_RUNTIME_SO="$(find "$BUILD" -type f -name libexecutorch_vulkan_shared_runtime.so -print -quit)"
LIBCXX_SHARED_SO="$(find "$ANDROID_NDK/toolchains/llvm/prebuilt" -type f \
  -path '*/aarch64-linux-android/libc++_shared.so' -print -quit)"
test -f "$SHARED_RUNTIME_SO" && test -f "$LIBCXX_SHARED_SO"
adb push "$SHARED_RUNTIME_SO" "$DEVICE_DIR/"
adb push "$LIBCXX_SHARED_SO" "$DEVICE_DIR/"
adb shell "cd $DEVICE_DIR && chmod 755 vgf_vulkan_shared_runner && \
  LD_LIBRARY_PATH=$DEVICE_DIR ./vgf_vulkan_shared_runner ."
```

Success prints shared-device identity, `PASS iteration=... max_abs_error=...`
and `PASS teardown cycle=...`. Each process performs two load/run/destroy cycles,
with three executions per cycle. The shared path checks that supported optional
integer-dot-product, subgroup, FP64 and KHR cooperative-matrix features are
reported enabled. Unsupported optional features remain optional.

## Regression modes and CI

```bash
python -m examples.arm.vgf_vulkan_shared.export \
  --output-dir vgf_vulkan_first_out --vulkan-first
"$RUNNER" vgf_vulkan_first_out --shared-vulkan-first
"$RUNNER" vgf_vulkan_shared_out --legacy
"$RUNNER" vgf_vulkan_first_out --legacy-vulkan-first
```

`--vulkan-first` exports `Vulkan -> VGF` with the same input/output shapes.
The runner's `*-vulkan-first` modes initialize ET-VK before any VGF registration
or probe. The legacy modes pass **no sharing options**. Invoke modes separately
so an earlier test cannot mask singleton-initialization bugs. `test_runtime_vgf`
builds and executes all four combinations through the Vulkan validation wrapper.
