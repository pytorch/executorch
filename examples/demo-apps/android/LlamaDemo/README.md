# Welcome to the ExecuTorch LLaMA Android Demo App

This app demonstrates the use of the LLaMA chat app demostrating local inference use case with ExecuTorch.

::::{grid} 2
:::{grid-item-card}  What you will learn
:class-card: card-prerequisites
* How to set up a build target for Android arm64-v8a
* How to build the required ExecuTorch runtime, LLaMA runner, and JNI wrapper for Android
* How to build the app with required JNI library
:::
:::{grid-item-card} Prerequisites
:class-card: card-prerequisites
* Refer to [Setting up ExecuTorch](https://pytorch.org/executorch/stable/getting-started-setup) to set up the repo and dev environment.
* Download and install [Android Studio and SDK](https://developer.android.com/studio).
* Supported Host OS: CentOS, macOS Sonoma on Apple Silicon.
:::
::::

```{note}
This demo app and tutorial has only been validated with arm64-v8a [ABI](https://developer.android.com/ndk/guides/abis), with NDK 25.
```

## Getting models
Please refer to the LLaMA tutorial to export the model.

Once you have the model, you need to push it to your device:
```bash
adb shell mkdir -p /data/local/tmp/llama
adb push model.pte /data/local/tmp/llama
adb push tokenizer.bin /data/local/tmp/llama
```

```{note}
The demo app searches in `/data/local/tmp/llama` for .pte and .bin files as LLAMA model and tokenizer.
```

## Build JNI library
1. Open a terminal window and navigate to the root directory of the `executorch`.
2. Set the following environment variables:
```
export ANDROID_NDK=<path_to_android_ndk>
export ANDROID_ABI=arm64-v8a
```
3. Create a new directory for the CMake build output:
```
mkdir cmake-out
```
4. Run the following command to configure the CMake build:
```
# Build the core ExecuTorch runtime library
cmake . -DCMAKE_INSTALL_PREFIX=cmake-out \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="${ANDROID_ABI}" \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_OPTIMIZED=ON \
  -Bcmake-out

cmake --build cmake-out -j16 --target install

# Build the llama2 runner library and custom ops
cmake examples/models/llama2 \
         -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
         -DANDROID_ABI="$ANDROID_ABI" \
         -DCMAKE_INSTALL_PREFIX=cmake-out \
         -Bcmake-out/examples/models/llama2

cmake --build cmake-out/examples/models/llama2 -j16

# Build the Android JNI library
cmake extension/android \
  -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI="${ANDROID_ABI}" \
  -DCMAKE_INSTALL_PREFIX=cmake-out \
  -DEXECUTORCH_BUILD_LLAMA_JNI=ON \
  -Bcmake-out/extension/android

cmake --build cmake-out/extension/android -j16
```

5. Copy the built library to your app:
```
JNI_LIBS_PATH="examples/demo-apps/android/LlamaDemo/app/src/main/jniLibs"
mkdir -p "${JNI_LIBS_PATH}/${ANDROID_ABI}"
cp cmake-out/extension/android/libexecutorch_llama_jni.so "${JNI_LIBS_PATH}/${ANDROID_ABI}/"
```

## Build Java app
1. Open Android Studio and select "Open an existing Android Studio project" to open examples/demo-apps/android/LlamaDemo.
2. Run the app (^R).

On the phone or emulator, you can try running the model:
<img src="_static/img/android_llama_app.png" alt="Android LLaMA App" /><br>

## Takeaways
Through this tutorial we've learnt how to build the ExecuTorch LLAMA library with XNNPACK backend, and expose it to JNI layer to build the Android app.

## Reporting Issues
If you encountered any bugs or issues following this tutorial please file a bug/issue here on [Github](https://github.com/pytorch/executorch/issues/new).
