# Building and Running ExecuTorch with the Vulkan Backend

The [ExecuTorch Vulkan Delegate](./native-delegates-executorch-vulkan-delegate.md)
is a native GPU delegate for ExecuTorch.

<!----This will show a grid card on the page----->
::::{grid} 2
:::{grid-item-card}  What you will learn in this tutorial:
:class-card: card-content
* How to export the Llama3.2-1B parameter model with partial GPU delegation
* How to execute the partially delegated model on Android
:::
:::{grid-item-card}  Prerequisites:
:class-card: card-prerequisites
* Follow [**Setting up ExecuTorch**](./getting-started-setup.md)
* It is also recommended that you read through [**ExecuTorch Vulkan Delegate**](./native-delegates-executorch-vulkan-delegate.md) and follow the example in that page
:::
::::

## Prerequisites

Note that all the steps below should be performed from the ExecuTorch repository
root directory, and assumes that you have gone through the steps of setting up
ExecuTorch.

It is also assumed that the Android NDK and Android SDK is installed, and the
following environment examples are set.

```shell
export ANDROID_NDK=<path_to_ndk>
# Select an appropriate Android ABI for your device
export ANDROID_ABI=arm64-v8a
# All subsequent commands should be performed from ExecuTorch repo root
cd <path_to_executorch_root>
# Make sure adb works
adb --version
```

## Lowering the Llama3.2-1B model to Vulkan

::::{note}
The resultant model will only be partially delegated to the Vulkan backend. In
particular, only binary arithmetic operators (`aten.add`, `aten.sub`,
`aten.mul`, `aten.div`), matrix multiplication operators (`aten.mm`, `aten.bmm`),
and linear layers (`aten.linear`) will be executed on the GPU via the Vulkan
delegate. The rest of the model will be executed using Portable operators.

Operator support for LLaMA models is currently in active development; please
check out the `main` branch of the ExecuTorch repo for the latest capabilities.
::::

First, obtain the `consolidated.00.pth`, `params.json` and `tokenizer.model`
files for the `Llama3.2-1B` model from the [Llama website](https://www.llama.com/llama-downloads/).

Once the files have been downloaded, the `export_llama` script can be used to
partially lower the Llama model to Vulkan.

```shell
# The files will usually be downloaded to ~/.llama
python -m examples.models.llama2.export_llama \
  --disable_dynamic_shape --vulkan -kv --use_sdpa_with_kv_cache -d fp32 \
  -c ~/.llama/checkpoints/Llama3.2-1B/consolidated.00.pth \
  -p ~/.llama/checkpoints/Llama3.2-1B/params.json \
  --metadata '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}'
```

A `vulkan_llama2.pte` file should have been created as a result of running the
script.

Push the tokenizer binary and `vulkan_llama2.pte` onto your Android device:

```shell
adb push ~/.llama/tokenizer.model /data/local/tmp/
adb push vulkan_llama2.pte /data/local/tmp/
```

## Build and Run the LLaMA runner binary on Android

First, build and install ExecuTorch libraries, then build the LLaMA runner
binary using the Android NDK toolchain.

```shell
(rm -rf cmake-android-out && \
  cmake . -DCMAKE_INSTALL_PREFIX=cmake-android-out \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=$ANDROID_ABI \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_VULKAN=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
    -DPYTHON_EXECUTABLE=python \
    -Bcmake-android-out && \
  cmake --build cmake-android-out -j16 --target install)

# Build LLaMA Runner library
(rm -rf cmake-android-out/examples/models/llama2 && \
  cmake examples/models/llama2 \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=$ANDROID_ABI \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
    -DCMAKE_INSTALL_PREFIX=cmake-android-out \
    -DPYTHON_EXECUTABLE=python \
    -Bcmake-android-out/examples/models/llama2 && \
  cmake --build cmake-android-out/examples/models/llama2 -j16)
```

Finally, push and run the llama runner binary on your Android device. Note that
your device must have sufficient GPU memory to execute the model.

```shell
adb push cmake-android-out/examples/models/llama2/llama_main /data/local/tmp/llama_main

adb shell /data/local/tmp/llama_main \
    --model_path=/data/local/tmp/vulkan_llama2.pte \
    --tokenizer_path=/data/local/tmp/tokenizer.model \
    --prompt "Hello"
```

Note that currently model inference will be very slow due to the high amount of
delegate blobs in the lowered graph, which requires a transfer to and from the
GPU for each sub graph. Performance is expected to improve drastically as more
of the model can be lowered to the Vulkan delegate, and techniques such as
quantization are supported.
