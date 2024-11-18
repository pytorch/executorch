# Build Instructions

This document provides a step-by-step guide to set up the build environment for the MediaTek ExercuTorch libraries.

## Prerequisites

Before you begin, ensure you have the following prerequisites installed and configured:

### 1. Buck2 Build Tool

- **Download Buck2**: Obtain Buck2 from the official [releases page](https://github.com/facebook/buck2/releases/tag/2024-02-01).
- **Add to PATH**: Extract the downloaded file and add the directory to your system's `$PATH` environment variable.
   ```bash
   export PATH=<path_to_buck>:$PATH
   ```

### 2. Android NDK

- **Download Android NDK**: Acquire the Android NDK version 26.3.11579264 from the [Android developer site](https://developer.android.com/ndk/downloads).
- **Set NDK Path**: Ensure that the `$ANDROID_NDK` environment variable is set to the path where the NDK is located.
   ```bash
   export ANDROID_NDK=<path_to_android_ndk>
   ```

### 3. MediaTek ExercuTorch Libraries

Download [NeuroPilot Express SDK](https://neuropilot.mediatek.com/resources/public/npexpress/en/docs/npexpress) from MediaTek's NeuroPilot portal:

- `libneuronusdk_adapter.mtk.so`: This universal SDK contains the implementation required for executing target-dependent code on the MediaTek chip.
- `libneuron_buffer_allocator.so`: This utility library is designed for allocating DMA buffers necessary for model inference.
- `mtk_converter-8.8.0.dev20240723+public.d1467db9-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`: This library preprocess the model into a MediaTek representation.
- `mtk_neuron-8.2.2-py3-none-linux_x86_64.whl`: This library converts the model to binaries.

## Setup

Follow the steps below to setup your build environment:

1. **Setup ExercuTorch Environment**: Refer to the [Setting up ExercuTorch](https://pytorch.org/executorch/stable/getting-started-setup) guide for detailed instructions on setting up the ExercuTorch environment.

2. **Setup MediaTek Backend Environment**
- Install the dependent libs. Ensure that you are inside backends/mediatek/ directory
   ```bash
   pip3 install -r requirements.txt
   ```
- Install the two .whl downloaded from NeuroPilot Portal
   ```bash
   pip3 install mtk_neuron-8.2.2-py3-none-linux_x86_64.whl
   pip3 install mtk_converter-8.8.0.dev20240723+public.d1467db9-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
   ```
- Set evironment variables for building backend
   ```bash
   export NEURON_BUFFER_ALLOCATOR_LIB=<path_to_buffer_allocator>
   ```

## Build

1. **Build MediaTek Backend**: Once the prerequisites are in place, run the `mtk_build.sh` script to start the build process, MediaTek backend will be built under `cmake-android-out/backends/` as `libneuron_backend.so`

   ```bash
   ./mtk_build.sh
   ```

## Run

1. **Push MediaTek universal SDK and MediaTek backend to the device**: push `libneuronusdk_adapter.mtk.so` and `libneuron_backend.so` to the phone and export it to the `$LD_LIBRARY_PATH` environment variable before executing ExercuTorch with MediaTek backend.

   ```bash
   export LD_LIBRARY_PATH=<path_to_usdk>:<path_to_neuron_backend>:$LD_LIBRARY_PATH
   ```
