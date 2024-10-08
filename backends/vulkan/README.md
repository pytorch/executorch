# ExecuTorch Vulkan Delegate

The ExecuTorch Vulkan delegate is a native GPU delegate for ExecuTorch that is
built on top of the cross-platform Vulkan GPU API standard. It is primarily
designed to leverage the GPU to accelerate model inference on Android devices,
but can be used on any platform that supports an implementation of Vulkan:
laptops, servers, and edge devices.

::::{note}
The Vulkan delegate is currently under active development, and its components
are subject to change.
::::

## What is Vulkan?

Vulkan is a low-level GPU API specification developed as a successor to OpenGL.
It is designed to offer developers more explicit control over GPUs compared to
previous specifications in order to reduce overhead and maximize the
capabilities of the modern graphics hardware.

Vulkan has been widely adopted among GPU vendors, and most modern GPUs (both
desktop and mobile) in the market support Vulkan. Vulkan is also included in
Android from Android 7.0 onwards.

**Note that Vulkan is a GPU API, not a GPU Math Library**. That is to say it
provides a way to execute compute and graphics operations on a GPU, but does not
come with a built-in library of performant compute kernels.

## The Vulkan Compute Library

The ExecuTorch Vulkan Delegate is a wrapper around a standalone runtime known as
the **Vulkan Compute Library**. The aim of the Vulkan Compute Library is to
provide GPU implementations for PyTorch operators via GLSL compute shaders.

The Vulkan Compute Library is a fork/iteration of the [PyTorch Vulkan Backend](https://pytorch.org/tutorials/prototype/vulkan_workflow.html).
The core components of the PyTorch Vulkan backend were forked into ExecuTorch
and adapted for an AOT graph-mode style of model inference (as opposed to
PyTorch which adopted an eager execution style of model inference).

The components of the Vulkan Compute Library are contained in the
`executorch/backends/vulkan/runtime/` directory. The core components are listed
and described below:

```
runtime/
├── api/ .................... Wrapper API around Vulkan to manage Vulkan objects
└── graph/ .................. ComputeGraph class which implements graph mode inference
    └── ops/ ................ Base directory for operator implementations
        ├── glsl/ ........... GLSL compute shaders
        │   ├── *.glsl
        │   └── conv2d.glsl
        └── impl/ ........... C++ code to dispatch GPU compute shaders
            ├── *.cpp
            └── Conv2d.cpp
```

## Features

The Vulkan delegate currently supports the following features:

* **Memory Planning**
  * Intermediate tensors whose lifetimes do not overlap will share memory allocations. This reduces the peak memory usage of model inference.
* **Capability Based Partitioning**:
  * A graph can be partially lowered to the Vulkan delegate via a partitioner, which will identify nodes (i.e. operators) that are supported by the Vulkan delegate and lower only supported subgraphs
* **Support for upper-bound dynamic shapes**:
  * Tensors can change shape between inferences as long as its current shape is smaller than the bounds specified during lowering

In addition to increasing operator coverage, the following features are
currently in development:

* **Quantization Support**
  * We are currently working on support for 8-bit dynamic quantization, with plans to extend to other quantization schemes in the future.
* **Memory Layout Management**
  * Memory layout is an important factor to optimizing performance. We plan to introduce graph passes to introduce memory layout transitions throughout a graph to optimize memory-layout sensitive operators such as Convolution and Matrix Multiplication.
* **Selective Build**
  * We plan to make it possible to control build size by selecting which operators/shaders you want to build with

## End to End Example

To further understand the features of the Vulkan Delegate and how to use it,
consider the following end to end example with a simple single operator model.

### Compile and lower a model to the Vulkan Delegate

Assuming ExecuTorch has been set up and installed, the following script can be
used to produce a lowered MobileNet V2 model as `vulkan_mobilenetv2.pte`.

Once ExecuTorch has been set up and installed, the following script can be used
to generate a simple model and lower it to the Vulkan delegate.

```
# Note: this script is the same as the script from the "Setting up ExecuTorch"
# page, with one minor addition to lower to the Vulkan backend.
import torch
from torch.export import export
from executorch.exir import to_edge

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner

# Start with a PyTorch model that adds two input tensors (matrices)
class Add(torch.nn.Module):
  def __init__(self):
    super(Add, self).__init__()

  def forward(self, x: torch.Tensor, y: torch.Tensor):
      return x + y

# 1. torch.export: Defines the program with the ATen operator set.
aten_dialect = export(Add(), (torch.ones(1), torch.ones(1)))

# 2. to_edge: Make optimizations for Edge devices
edge_program = to_edge(aten_dialect)
# 2.1 Lower to the Vulkan backend
edge_program = edge_program.to_backend(VulkanPartitioner())

# 3. to_executorch: Convert the graph to an ExecuTorch program
executorch_program = edge_program.to_executorch()

# 4. Save the compiled .pte program
with open("vk_add.pte", "wb") as file:
    file.write(executorch_program.buffer)
```

Like other ExecuTorch delegates, a model can be lowered to the Vulkan Delegate
using the `to_backend()` API. The Vulkan Delegate implements the
`VulkanPartitioner` class which identifies nodes (i.e. operators) in the graph
that are supported by the Vulkan delegate, and separates compatible sections of
the model to be executed on the GPU.

This means the a model can be lowered to the Vulkan delegate even if it contains
some unsupported operators. This will just mean that only parts of the graph
will be executed on the GPU.


::::{note}
The [supported ops list](https://github.com/pytorch/executorch/blob/main/backends/vulkan/partitioner/supported_ops.py)
Vulkan partitioner code can be inspected to examine which ops are currently
implemented in the Vulkan delegate.
::::

### Build Vulkan Delegate libraries

The easiest way to build and test the Vulkan Delegate is to build for Android
and test on a local Android device. Android devices have built in support for
Vulkan, and the Android NDK ships with a GLSL compiler which is needed to
compile the Vulkan Compute Library's GLSL compute shaders.

The Vulkan Delegate libraries can be built by setting `-DEXECUTORCH_BUILD_VULKAN=ON`
when building with CMake.

First, make sure that you have the Android NDK installed; any NDK version past
NDK r19c should work. Note that the examples in this doc have been validated with
NDK r25. The Android SDK should also be installed so that you have access to `adb`.

The instructions in this page assumes that the following environment variables
are set.

```shell
export ANDROID_NDK=<path_to_ndk>
# Select the appropriate Android ABI for your device
export ANDROID_ABI=arm64-v8a
# All subsequent commands should be performed from ExecuTorch repo root
cd <path_to_executorch_root>
# Make sure adb works
adb --version
```

To build and install ExecuTorch libraries (for Android) with the Vulkan
Delegate:

```shell
# From executorch root directory
(rm -rf cmake-android-out && \
  pp cmake . -DCMAKE_INSTALL_PREFIX=cmake-android-out \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=$ANDROID_ABI \
    -DEXECUTORCH_BUILD_VULKAN=ON \
    -DPYTHON_EXECUTABLE=python \
    -Bcmake-android-out && \
  cmake --build cmake-android-out -j16 --target install)
```

### Run the Vulkan model on device

::::{note}
Since operator support is currently limited, only binary arithmetic operators
will run on the GPU. Expect inference to be slow as the majority of operators
are being executed via Portable operators.
::::

Now, the partially delegated model can be executed (partially) on your device's
GPU!

```shell
# Build a model runner binary linked with the Vulkan delegate libs
cmake --build cmake-android-out --target vulkan_executor_runner -j32

# Push model to device
adb push vk_add.pte /data/local/tmp/vk_add.pte
# Push binary to device
adb push cmake-android-out/backends/vulkan/vulkan_executor_runner /data/local/tmp/runner_bin

# Run the model
adb shell /data/local/tmp/runner_bin --model_path /data/local/tmp/vk_add.pte
```
