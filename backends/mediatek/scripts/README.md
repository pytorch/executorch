# Build Instructions

This document provides a step-by-step guide to set up the build environment for the MediaTek ExercuTorch libraries.

## Prerequisites

Before you begin, ensure you have the following prerequisites installed and configured:

### 1. Buck2 Build Tool

- **Download Buck2**: Obtain Buck2 from the official [releases page](https://github.com/facebook/buck2/releases/tag/2024-02-01).
- **Create Buck2 executable**: Extract the downloaded file and create an executable buck2.
```bash
zstd -cdq "<downloaded_buck2_file>.zst" > "<path_to_store_buck2>/buck2" && chmod +x "<path_to_store_buck2>/buck2"
```
- **Set BUCK2 Path**: Ensure that the `$BUCK2` environment variable is set to the path to point to buck2.
```bash
export BUCK2=<path_to_store_buck2>/buck2
```

### 2. Android NDK

- **Download Android NDK**: Acquire the Android NDK from the [Android developer site](https://developer.android.com/ndk/downloads).
- **Set NDK Path**: Ensure that the `$ANDROID_NDK` environment variable is set to the path where the NDK is located.
```bash
export ANDROID_NDK=<path_to_android_ndk>
```

### 3. MediaTek ExercuTorch Libraries

Download the following libraries from MediaTek's NeuroPilot portal (link to be added):

- `libneuronusdk_adapter.mtk.so`: This universal SDK contains the implementation required for executing target-dependent code on the MediaTek chip.
- `libneuron_buffer_allocator.so`: This utility library is designed for allocating DMA buffers necessary for model inference.
  - **Set NEURON LIB Path**: Add the pointer to the file by exporting to the `$NEURON_BUFFER_ALLOCATOR_LIB` environment variable.
```bash
export NEURON_BUFFER_ALLOCATOR_LIB=<path_to_buffer_allocator/libneuron_buffer_allocator.so>
```

## Setup

Follow the steps below to set up your build environment:

1. **ExercuTorch Official Tutorial**: Refer to the [Setting up ExercuTorch](https://pytorch.org/executorch/stable/getting-started-setup) guide for detailed instructions on setting up the ExercuTorch environment.

2. **Build Script**: Once the prerequisites are in place, run the `mtk_build.sh` script to start the build process.

   ```bash
   ./mtk_build.sh
   ```
3. **Push MediaTek universal SDK to the device**: push libneuronusdk_adapter.mtk.so to the phone and export it to the `$LD_LIBRARY_PATH` environment variable before executing ExercuTorch with MediaTek backend.

   ```bash
   export LD_LIBRARY_PATH=<path_to_usdk>:$LD_LIBRARY_PATH
   ```
