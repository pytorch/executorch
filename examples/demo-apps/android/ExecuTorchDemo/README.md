# Android Demo App: ExecuTorch Setup

This is forked from [PyTorch android demo app](https://github.com/pytorch/android-demo-app)

This guide explains how to setup ExecuTorch for Android using a demo app. The app employs a DeepLab v3 model for image segmentation tasks. Models are exported to ExecuTorch using XNNPACK FP32 backend.

## Pre-setup

1. Refer to [Setting up ExecuTorch](../../../../docs/website/docs/tutorials/00_setting_up_executorch.md) to set up the repo and dev environment.

2. Download and install [Android Studio and SDK](https://developer.android.com/studio).

## ExecuTorch Configuration

Configure the libraries for Android:

1. Configure a library with XNNPACK backend only

> **Note**: This demo app and tutorial has only been validated with arm64-v8a [ABI](https://developer.android.com/ndk/guides/abis).

```bash
export ANDROID_NDK=<path-to-android-ndk>

rm -rf cmake-out && mkdir cmake-out && cd cmake-out
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DBUCK2=/tmp/buck2 \
    -DEXECUTORCH_BUILD_ANDROID_DEMO_APP_JNI=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_FLATC=OFF \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON
```

When we set `EXECUTORCH_BUILD_XNNPACK=ON`, we will build the target [`xnn_executor_runner_lib`](../../../../backends/xnnpack/CMakeLists.txt) which in turn is linked into libexecutorchdemo via [CMake](../jni/CMakeLists.txt).

`libexecutorchdemo.so` wraps up the required XNNPACK Backend runtime library from `xnn_executor_runner_lib`, and adds an additional JNI layer using fbjni. This is later exposed to Java app.

2. *Optional:* Configure a library with XNNPACK and [Qualcomm HTP backend](../../../../backends/qualcomm/README.md)

> **Note**: Qualcomm SDK is required for this step.

```bash
rm -rf cmake-out && mkdir cmake-out && cd cmake-out
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=/path/to/ndk/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DBUCK2=/tmp/buck2 \
    -DFLATC_EXECUTABLE=$(realpath ../third-party/flatbuffers/cmake-out/flatc) \
    -DEXECUTORCH_BUILD_ANDROID_DEMO_APP_JNI=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_FLATC=OFF \
    -DEXECUTORCH_BUILD_QNN=ON \
    -DQNN_SDK_ROOT=/path/to/qnn/sdk \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON
```

Similar to the previous XNNPACK library, with this setup, we compile `libexecutorchdemo.so` but it adds an additional static library `qnn_executorch_backend` which wraps up Qualcomm HTP runtime library and registers the Qualcomm HTP backend. This is later exposed to Java app.

`qnn_executorch_backend` is built when we turn on CMake option `EXECUTORCH_BUILD_QNN`. It will include the [CMakeLists.txt](../../../../backends/qualcomm/CMakeLists.txt) from backends/qualcomm where we `add_library(qnn_executorch_backend STATIC)`.

## Building and Copying Libraries

1. Build the libraries:

```bash
cmake --build . -j16
```

2. Copy the libraries to the appropriate location to link against them:

Navigate to the build artifacts directory:

```bash
mkdir -p ../examples/demo-apps/android/ExecuTorchDemo/app/src/main/jniLibs/arm64-v8a
```

Copy the core libraries:

```bash
cp ./examples/demo-apps/android/jni/libexecutorchdemo.so \
   ../examples/demo-apps/android/ExecuTorchDemo/app/src/main/jniLibs/arm64-v8a
```

Later, this shared library will be loaded by `NativePeer.java` in Java code.

3. *Qualcomm HTP Only:* Copy Qualcomm HTP runtime library

```bash
cp libQnnHtp.so libQnnHtpV69Skel.so libQnnHtpStub.so libQnnSystem.so \
   ../examples/demo-apps/android/ExecuTorchDemo/app/src/main/jniLibs/arm64-v8a
```

4. Return to the `executorch` directory:

```bash
cd ..
```

## Model Download and Bundling

> **Note**: Please refer to [XNNPACK backend](../../../backend/README.md) and [Qualcomm backend](../../../../backends/qualcomm/README.md) for the full export tutorial on backends.

1. Export a DeepLab v3 model and Inception v4 model backed with XNNPACK delegate and bundle it with
   the app:

```bash
export FLATC_EXECUTABLE=$(realpath third-party/flatbuffers/cmake-out/flatc)
python3 -m examples.xnnpack.aot_compiler --model_name="dl3" --delegate
mkdir -p examples/demo-apps/android/ExecuTorchDemo/app/src/main/assets/
cp dl3_xnnpack_fp32.pte examples/demo-apps/android/ExecuTorchDemo/app/src/main/assets/
```

2. *Qualcomm HTP Only:* Copy HTP delegation models to the app:

```bash
cp dlv3_qnn.pte examples/demo-apps/android/ExecuTorchDemo/app/src/main/assets/
```

## Final Steps

1. Open the project `examples/demo-apps/android/ExecuTorchDemo` with Android Studio.

2. [Run](https://developer.android.com/studio/run) the app (^R)
