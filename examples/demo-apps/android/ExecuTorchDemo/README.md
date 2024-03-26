# Building an ExecuTorch Android Demo App

This is forked from [PyTorch Android Demo App](https://github.com/pytorch/android-demo-app).

This guide explains how to setup ExecuTorch for Android using a demo app. The app employs a [DeepLab v3](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/) model for image segmentation tasks. Models are exported to ExecuTorch using [XNNPACK FP32 backend](tutorial-xnnpack-delegate-lowering.md).

::::{grid} 2
:::{grid-item-card}  What you will learn
:class-card: card-prerequisites
* How to set up a build target for Android arm64-v8a
* How to build the required ExecuTorch runtime with JNI wrapper for Android
* How to build the app with required JNI library and model file
:::

:::{grid-item-card} Prerequisites
:class-card: card-prerequisites
* Refer to [Setting up ExecuTorch](https://pytorch.org/executorch/stable/getting-started-setup) to set up the repo and dev environment.
* Download and install [Android Studio and SDK](https://developer.android.com/studio).
* Supported Host OS: CentOS, macOS Ventura (M1/x86_64). See below for Qualcomm HTP specific requirements.
* *Qualcomm HTP Only[^1]:* To build and run on Qualcomm's AI Engine Direct, please follow [Building and Running ExecuTorch with Qualcomm AI Engine Direct Backend](build-run-qualcomm-ai-engine-direct-backend.md) for hardware and software pre-requisites.
:::
::::

[^1]: This section applies only if Qualcomm HTP Backend is needed in the app. Same applies to sections with title`Qualcomm Hexagon NPU`.

```{note}
This demo app and tutorial has only been validated with arm64-v8a [ABI](https://developer.android.com/ndk/guides/abis).
```


## Build

### Ahead-Of-Time

We generate the model file for the ExecuTorch runtime in Android Demo App.

#### XNNPACK Delegation

For delegating DeepLab v3 to XNNPACK backend, please do the following to export the model:

```bash
export FLATC_EXECUTABLE=$(realpath third-party/flatbuffers/cmake-out/flatc)
python3 -m examples.xnnpack.aot_compiler --model_name="dl3" --delegate
mkdir -p examples/demo-apps/android/ExecuTorchDemo/app/src/main/assets/
cp dl3_xnnpack_fp32.pte examples/demo-apps/android/ExecuTorchDemo/app/src/main/assets/
```

For more detailed tutorial of lowering to XNNPACK, please see [XNNPACK backend](tutorial-xnnpack-delegate-lowering.md).

#### Qualcomm Hexagon NPU

For delegating to Qualcomm Hexagon NPU, please follow the tutorial [here](build-run-qualcomm-ai-engine-direct-backend.md).

After generating the model, copy the model to `assets` directory.

```bash
python -m examples.qualcomm.scripts.deeplab_v3 -b build_android -m SM8550 -s <adb_connected_device_serial>
cp deeplab_v3/dlv3_qnn.pte examples/demo-apps/android/ExecuTorchDemo/app/src/main/assets/
```

### Runtime

We build the required ExecuTorch runtime library to run the model.

#### Java helper classes

Note: This is an ad-hoc solution. We will publish a formal Java package when it is ready. However, for now we need to copy sources from extension/android/src/main/java/org/pytorch/executorch.

```
mkdir -p examples/demo-apps/android/ExecuTorchDemo/app/src/main/java/com/example/executorchdemo/executor
cp extension/android/src/main/java/org/pytorch/executorch/*.java examples/demo-apps/android/ExecuTorchDemo/app/src/main/java/com/example/executorchdemo/executor
```

#### XNNPACK

1. Configure the CMake target for the library with XNNPACK backend:

```bash
export ANDROID_NDK=<path-to-android-ndk>
export BUCK2=/tmp/buck2 # Or your buck path

rm -rf cmake-out && mkdir cmake-out && cd cmake-out
cmake .. -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
        -DANDROID_ABI=arm64-v8a \
        -DBUCK2=$BUCK \
        -DEXECUTORCH_BUILD_XNNPACK=ON \
        -DEXECUTORCH_BUILD_FLATC=OFF \
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
        -DFLATC_EXECUTABLE=$FLATC_EXECUTABLE \
        -DEXECUTORCH_BUILD_ANDROID_JNI=ON \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON
```

When we set `EXECUTORCH_BUILD_XNNPACK=ON`, we will build the target [`xnnpack_backend`](https://github.com/pytorch/executorch/blob/main/backends/xnnpack/CMakeLists.txt) which in turn is linked into libexecutorch_jni via [CMake](https://github.com/pytorch/executorch/blob/main/examples/demo-apps/android/jni/CMakeLists.txt).

`libexecutorch_jni.so` wraps up the required XNNPACK Backend runtime library from `xnnpack_backend`, and adds an additional JNI layer using fbjni. This is later exposed to Java app.

2. Build the libraries:

```bash
cmake --build . -j16
```

#### Qualcomm Hexagon NPU

1. Configure the CMake target for the library with Qualcomm Hexagon NPU (HTP) backend (XNNPACK also included):

```bash
export ANDROID_NDK=<path-to-android-ndk>
export QNN_SDK=<path-to-qnn-sdk>

rm -rf cmake-out && mkdir cmake-out && cd cmake-out
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DBUCK2=/tmp/buck2 \
    -DEXECUTORCH_BUILD_ANDROID_JNI=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_FLATC=OFF \
    -DEXECUTORCH_BUILD_QNN=ON \
    -DQNN_SDK_ROOT=$QNN_SDK \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON
```
Similar to the XNNPACK library, with this setup, we compile `libexecutorch_jni.so` but it adds an additional static library `qnn_executorch_backend` which wraps up Qualcomm HTP runtime library and registers the Qualcomm HTP backend. This is later exposed to Java app.

`qnn_executorch_backend` is built when we turn on CMake option `EXECUTORCH_BUILD_QNN`. It will include the [CMakeLists.txt](https://github.com/pytorch/executorch/blob/main/backends/qualcomm/CMakeLists.txt) from backends/qualcomm where we `add_library(qnn_executorch_backend STATIC)`.

2. Build the libraries:

```bash
cmake --build . -j16
```

## Deploying on Device via Demo App

### Steps for Deploying Model via XNNPACK

```bash
mkdir -p ../examples/demo-apps/android/ExecuTorchDemo/app/src/main/jniLibs/arm64-v8a
```

Copy the core libraries:

```bash
cp ./examples/demo-apps/android/jni/libexecutorch_jni.so \
   ../examples/demo-apps/android/ExecuTorchDemo/app/src/main/jniLibs/arm64-v8a/libexecutorch.so
```

This allows the Android app to load ExecuTorch runtime with XNNPACK backend as a JNI library. Later, this shared library will be loaded by `NativePeer.java` in Java code.

### Steps for Deploying Model via Qualcomm's AI Engine Direct

```bash
mkdir -p ../examples/demo-apps/android/ExecuTorchDemo/app/src/main/jniLibs/arm64-v8a
```

We need to push some additional Qualcomm HTP backend libraries to the app. Please refer to [Qualcomm docs](build-run-qualcomm-ai-engine-direct-backend.md) here.

```bash
cp ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV69Skel.so ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpStub.so ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnSystem.so \
   ../examples/demo-apps/android/ExecuTorchDemo/app/src/main/jniLibs/arm64-v8a
```

Copy the core libraries:

```bash
cp ./examples/demo-apps/android/jni/libexecutorch_jni.so \
   ../examples/demo-apps/android/ExecuTorchDemo/app/src/main/jniLibs/arm64-v8a/libexecutorch.so
```

## Running the App

1. Open the project `examples/demo-apps/android/ExecuTorchDemo` with Android Studio.

2. [Run](https://developer.android.com/studio/run) the app (^R).

<img src="_static/img/android_studio.png" alt="Android Studio View" /><br>

On the phone or emulator, you can try running the model:
<img src="_static/img/android_demo_run.png" alt="Android Demo" /><br>

## Takeaways
Through this tutorial we've learnt how to build the ExecuTorch runtime library with XNNPACK (or Qualcomm HTP) backend, and expose it to JNI layer to build the Android app running segmentation model.

## Reporting Issues

If you encountered any bugs or issues following this tutorial please file a bug/issue here on [Github](https://github.com/pytorch/executorch/issues/new).
