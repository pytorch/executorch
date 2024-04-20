# Building and Running ExecuTorch with the Vulkan Backend

The [ExecuTorch Vulkan Delegate](./native-delegates-executorch-vulkan-delegate.md)
is a native GPU delegate for ExecuTorch.

<!----This will show a grid card on the page----->
::::{grid} 2
:::{grid-item-card}  What you will learn in this tutorial:
:class-card: card-content
* How to export the Stories 110M parameter model with partial GPU delegation
* How to execute the partially delegated model on Android
:::
:::{grid-item-card}  Prerequisites:
:class-card: card-prerequisites
* Follow [**Setting up ExecuTorch**](./getting-started-setup.md)
* Follow [**Setting up the ExecuTorch LLaMA Android Demo App**](./llm/llama-demo-android.md)
:::
::::

## Prerequisites

Note that all the steps below should be performed from the ExecuTorch repository
root directory, and assumes that you have gone through the steps of setting up
ExecuTorch.

You should also refer to the **Prerequisites** section of the [**Setting up the ExecuTorch LLaMA Android Demo App**](./llm/llama-demo-android.md)
Tutorial in order to install the specified versions of the Android NDK and the
Android SDK.

```shell
# Recommended version is Android NDK r25c.
export ANDROID_NDK=<path_to_ndk>
# Select an appropriate Android ABI
export ANDROID_ABI=arm64-v8a
# All subsequent commands should be performed from ExecuTorch repo root
cd <path_to_executorch_root>
# Make sure adb works
adb --version
```

## Lowering the Stories 110M model to Vulkan

::::{note}
The resultant model will only be partially delegated to the Vulkan backend. In
particular, only binary arithmetic operators (`aten.add`, `aten.sub`,
`aten.mul`, `aten.div`) and the matrix multiplication operator (`aten.mm`) will
be executed on the GPU via the Vulkan delegate. The rest of the model will be
executed using Portable operators. This is because the Vulkan delegate is still
early in development and currently has limited operator coverage.
::::

First, download `stories110M.pt` and `tokenizer.model` from Github:

```shell
wget "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.pt"
wget "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model"
```

Next, create the params file:

```shell
echo '{"dim": 768, "multiple_of": 32, "n_heads": 12, "n_layers": 12, "norm_eps": 1e-05, "vocab_size": 32000}' > params.json
```

Then, create a tokenizer binary file:

```shell
python -m examples.models.llama2.tokenizer.tokenizer -t tokenizer.model -o tokenizer.bin
```

Finally, export the `stories110M.pt` file into an ExecuTorch program:

```shell
python -m examples.models.llama2.export_llama -c stories110M.pt -p params.json --vulkan
```

A `vulkan_llama2.pte` file should have been created as a result of the last step.

Push the tokenizer binary and `vulkan_llama2.pte` onto your Android device:

```shell
adb mkdir /data/local/tmp/llama/
adb push tokenizer.bin /data/local/tmp/llama/
adb push vulkan_llama2.pte /data/local/tmp/llama/
```

## Build and Run the LLaMA runner binary on Android

First, build and install ExecuTorch libraries, then build the LLaMA runner
binary using the Android NDK toolchain.

```shell
(rm -rf cmake-android-out && \
  cmake . -DCMAKE_INSTALL_PREFIX=cmake-android-out \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=$ANDROID_ABI \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_VULKAN=ON \
    -DEXECUTORCH_BUILD_OPTIMIZED=ON \
    -DPYTHON_EXECUTABLE=python \
    -Bcmake-android-out && \
  cmake --build cmake-android-out -j16 --target install)

# Build LLaMA Runner library
(rm -rf cmake-android-out/examples/models/llama2 && \
  cmake examples/models/llama2 \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=$ANDROID_ABI \
    -DCMAKE_INSTALL_PREFIX=cmake-android-out \
    -DPYTHON_EXECUTABLE=python \
    -Bcmake-android-out/examples/models/llama2 && \
  cmake --build cmake-android-out/examples/models/llama2 -j16)
```

Finally, push and run the llama runner binary on your Android device.

```shell
adb push cmake-android-out/examples/models/llama2/llama_main /data/local/tmp/llama_main

adb shell /data/local/tmp/llama_main \
    --model_path=/data/local/tmp/llama/vulkan_llama2.pte \
    --tokenizer_path=/data/local/tmp/llama/tokenizer.bin \
    --prompt "hi" \--temperature=0
```

The following output will be produced:

```
hippo named Hippy lived in a big pond. Hippy was a very happy hippo. He liked to play...
```

## Running with the LLaMA Android Demo App

It is also possible to run the partially delegated Vulkan model inside the LLaMA
Android demo app.

First, make some modifications to the Android app setup script to make sure that
the Vulkan backend is built when building and installing ExecuTorch libraries:

```shell
# Run from executorch root directory. You can also edit this in a code editor
sed -i 's/-DEXECUTORCH_BUILD_XNNPACK=ON/-DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_VULKAN=ON/g' examples/demo-apps/android/LlamaDemo/setup.sh
```

Then, Follow the instructions at [**Setting up the ExecuTorch LLaMA Android Demo App**](./llm/llama-demo-android.md)
to build and run the demo application on your Android device. Once the app
starts up, you can load and run the `vulkan_llama2.pte` model with the app.
