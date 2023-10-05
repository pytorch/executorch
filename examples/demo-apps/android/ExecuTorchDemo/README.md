# Android Demo App: ExecuTorch Setup

This is forked from [PyTorch android demo app](https://github.com/pytorch/android-demo-app)

This guide explains how to setup ExecuTorch for Android using a demo app. The app employs a DeepLab v3 model for image segmentation tasks and Inception v3 model for image classification tasks. Models are exported to ExecuTorch using XNNPACK FP32 backend.

## Pre-setup

1. Download [Android Studio and SDK](https://developer.android.com/studio).

2. Install buck2 binary (using MacOS Apple Silicon build as example):
    ```bash
    curl -L -O https://github.com/facebook/buck2/releases/download/2023-07-18/buck2-aarch64-apple-darwin.zst
    pip3 install zstd
    zstd -cdq buck2-aarch64-apple-darwin.zst > /tmp/buck2 && chmod +x /tmp/buck2
    ```

3. Install and link [Cmake](cmake.org/download) to a system directory or `$PATH`:

   ```bash
   ln -s /Applications/CMake.app/Contents/bin/cmake /usr/bin/cmake
   ```

4. Clone ExecuTorch repository and update submodules:

   ```bash
   git clone https://github.com/pytorch/executorch.git
   cd executorch
   git submodule sync
   git submodule update --init
   ```

5. Verify Python 3.10+ (standard since MacOS 13.5) and `pip3` installation:

   ```bash
   which python3 pip
   python3 --version
   ```

6. Install PyTorch dependencies:

   ```bash
   ./install_requirements.sh
   ```

## Flatbuffers Compiler Setup

Run the following in the `flatbuffers` directory:

```bash
cd third-party/flatbuffers
rm -rf cmake-out && mkdir cmake-out && cd cmake-out
cmake .. && cmake --build . --target flatc
cd ../../..
```

## ExecuTorch Configuration

Configure the libraries for Android:

1. Configure a library with XNNPACK backend only

Note: This demo app and tutorial is only validated with arm64-v8a [ABI](https://developer.android.com/ndk/guides/abis).

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
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON
```

2. Configure a library with XNNPACK and Qualcomm HTP backend

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

## Building and Copying Libraries

1. Build the libraries:

```bash
cmake --build .
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

Then return to the `executorch` directory:

```bash
cd ..
```

## Model Download and Bundling

Note: Please refer to [XNNPACK backend](../../../backend/README.md) and [Qualcomm backend](../../../../backends/qualcomm/README.md) for the full export tutorial on backends.

1. Export a DeepLab v3 model and Inception v4 model backed with XNNPACK delegate and bundle it with
   the app:

```bash
export FLATC_EXECUTABLE=$(realpath third-party/flatbuffers/cmake-out/flatc)
python3 -m examples.backend.xnnpack_examples --model_name="dl3" --delegate
python3 -m examples.backend.xnnpack_examples --model_name="ic4" --delegate
cp dl3_xnnpack_fp32.pte ic4_xnnpack_fp32.pte examples/android_demo_apps/ExecuTorchDemo/app/src/main/assets/
```

## Final Steps

1. Open the project `examples/android_demo_apps/ExecuTorchDemo` with Android Studio.

2. [Run](https://developer.android.com/studio/run) the app (^R)
