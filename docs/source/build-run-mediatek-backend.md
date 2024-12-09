# Building and Running ExecuTorch with MediaTek Backend

MediaTek backend empowers ExecuTorch to speed up PyTorch models on edge devices that equips with MediaTek Neuron Processing Unit (NPU). This document offers a step-by-step guide to set up the build environment for the MediaTek ExecuTorch libraries.

::::{grid} 2
:::{grid-item-card}  What you will learn in this tutorial:
:class-card: card-prerequisites
* How to export and lower a PyTorch model ahead of time with ExecuTorch for MediaTek devices.
* How to build MediaTek backend and examples.
* How to deploy the exported models on device with ExecuTorch runtime.
:::
:::{grid-item-card}  Tutorials we recommend you complete before this:
:class-card: card-prerequisites
* [Introduction to ExecuTorch](intro-how-it-works.md)
* [Setting up ExecuTorch](getting-started-setup.md)
* [Building ExecuTorch with CMake](runtime-build-and-cross-compilation.md)
:::
::::


## Prerequisites (Hardware and Software)

### Host OS
- Linux operating system

### Supported Chips:
- MediaTek Dimensity 9300 (D9300)

### Software:

- [NeuroPilot Express SDK](https://neuropilot.mediatek.com/resources/public/npexpress/en/docs/npexpress) is a lightweight SDK for deploying AI applications on MediaTek SOC devices.

## Setting up your developer environment

Follow the steps below to setup your build environment:

1. **Setup ExecuTorch Environment**: Refer to the [Setting up ExecuTorch](https://pytorch.org/executorch/stable/getting-started-setup) guide for detailed instructions on setting up the ExecuTorch environment.

2. **Setup MediaTek Backend Environment**
- Install the dependent libs. Ensure that you are inside `backends/mediatek/` directory
   ```bash
   pip3 install -r requirements.txt
   ```
- Install the two .whl downloaded from NeuroPilot Portal
   ```bash
   pip3 install mtk_neuron-8.2.13-py3-none-linux_x86_64.whl
   pip3 install mtk_converter-8.9.1+public-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
   ```
- Set evironment variables for building backend
   ```bash
   export NEURON_BUFFER_ALLOCATOR_LIB=<path_to_buffer_allocator.so>
   ```

## Build

### Ahead of time:

**Exporting a PyTorch Model for MediaTek Backend**:
1. Lower and export the `.pte` file for on-device execution. The export script samples are povided under `example/mediatek/`. For example, the following commnad exports the `.pte` using the scripts provided.
```bash
cd executorch

./examples/mediatek/shell_scripts/export_oss.sh mobilenetv3
```

2. Find the `.pte` files under the directory named as same as the model.

### Runtime:

**Build MediaTek Backend for ExecuTorch Runtime**
1. Navigate to `backends/mediatek/scripts/` directory.

2. **Build MediaTek Backend**: Once the prerequisites are in place, run the `mtk_build.sh` script to start the build process:
   ```bash
   ./mtk_build.sh
   ```

3. MediaTek backend will be built under `cmake-android-out/backends/` as `libneuron_backend.so`.

**Build a runner to execute the model on the device**:
1. Build the runners and the backend by exedcuting the script:
```bash
./mtk_build_examples.sh
```

2. The runners will be built under `cmake-android-out/examples/`

## Deploying and running on a device

1. **Push MediaTek universal SDK and MediaTek backend to the device**: push `libneuronusdk_adapter.mtk.so` and `libneuron_backend.so` to the phone and export it to the `$LD_LIBRARY_PATH` environment variable before executing ExecuTorch with MediaTek backend.

   ```bash
   export LD_LIBRARY_PATH=<path_to_usdk>:<path_to_neuron_backend>:$LD_LIBRARY_PATH
   ```