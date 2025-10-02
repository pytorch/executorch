# MediaTek Backend on ExecuTorch
MediaTek backend empowers ExecuTorch to speed up PyTorch models on edge devices that equips with MediaTek Neuron Processing Unit (NPU). This document offers a step-by-step guide to set up the build environment for the MediaTek ExecuTorch libraries.

## Supported Chips

The examples provided in this repository are tested and supported on the following MediaTek chip:

- MediaTek Dimensity 9300 (D9300)
- MediaTek Dimensity 9400 (D9400)

## Build Instructions

### Prerequisites

Before you begin, ensure you have the following prerequisites installed and configured:

#### 1. Android NDK

- **Download Android NDK**: Acquire the Android NDK version 26.3.11579264 from the [Android developer site](https://developer.android.com/ndk/downloads).

#### 2. MediaTek ExecuTorch Libraries

To get started with MediaTek's ExecuTorch libraries, download the [NeuroPilot Express SDK](https://neuropilot.mediatek.com/resources/public/npexpress/en/docs/npexpress) from MediaTek's NeuroPilot portal. The SDK includes the following components:

- **`libneuronusdk_adapter.mtk.so`**: This universal SDK contains the implementation required for executing target-dependent code on the MediaTek chip.

- **`libneuron_buffer_allocator.so`**: This utility library is designed for allocating DMA buffers necessary for model inference.

- **`mtk_converter-8.13.0+public-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`**: This library preprocesses the model into a MediaTek representation.

- **`mtk_neuron-8.2.19-py3-none-linux_x86_64.whl`**: This library converts the model to binaries.

Additionally, make sure to copy `NeuronAdapter.h` to the following directory: `backends/mediatek/runtime/include/api/`.

### Setup

Follow the steps below to setup your build environment:

1. **Setup ExecuTorch Environment**: Refer to the [Setting up ExecuTorch](https://pytorch.org/executorch/main/getting-started-setup) guide for detailed instructions on setting up the ExecuTorch environment.

2. **Setup MediaTek Backend Environment**
- Install the dependent libs. Ensure that you are inside backends/mediatek/ directory
   ```bash
   pip3 install -r requirements.txt
   ```
- Install the two .whl downloaded from NeuroPilot Portal
   ```bash
   pip3 install mtk_neuron-8.2.19-py3-none-linux_x86_64.whl
   pip3 install mtk_converter-8.13.0+public-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
   ```

### Build
1. Copy `NeuronAdapter.h` to `backends/mediatek/runtime/include/api/`

2. Set NDK Path: Ensure that the `$ANDROID_NDK` environment variable is set to the path where the NDK is located.
   ```bash
   export ANDROID_NDK=<path_to_android_ndk>
   ```

3. Build the backend library `libneuron_backend.so`:
    ```bash
    cd backends/mediatek/scripts/
    ./mtk_build.sh
    ```
The output is `libneuron_backend.so` in `cmake-android-out/backends/mediatek/`.

### Run

1. Push `libneuron_backend.so`, `libneuronusdk_adapter.mtk.so` and `libneuron_buffer_allocator.so` to the device.
2. Set the library path before running ExecuTorch:
   ```bash
   export LD_LIBRARY_PATH=<path_to_neuron_backend>:<path_to_usdk>:<path_to_buffer_allocator>:$LD_LIBRARY_PATH
   ```

Please refer to `executorch/examples/mediatek/` for export and execution examples of various of models.
