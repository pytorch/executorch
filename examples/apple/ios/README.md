# iOS Demo App: ExecuTorch Setup

This guide explains how to setup ExecuTorch for iOS using a demo app. The app
employs a MobileNet v3 model (exported to ExecuTorch) to process live camera
images.

## Pre-setup

1. Install Xcode and Command Line Tools:

   ```bash
   xcode-select --install
   ```

2. Install `buck2` binary (using MacOS Apple Silicon build as example):

   ```bash
   curl -L -O https://github.com/facebook/buck2/releases/download/2023-07-18/buck2-aarch64-apple-darwin.zst
   pip3 install zstd
   zstd -cdq buck2-aarch64-apple-darwin.zst > /tmp/buck2 && chmod +x /tmp/buck2
   ```

3. Install [Cmake](cmake.org/download) and link it in a system directory or
   `$PATH`:

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

Configure the libraries for iOS:

```bash
rm -rf cmake-out && mkdir cmake-out && cd cmake-out
cmake .. -G Xcode \
    -DCMAKE_TOOLCHAIN_FILE=../third-party/pytorch/cmake/iOS.cmake \
    -DBUCK2=/tmp/buck2 \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DFLATC_EXECUTABLE=$(realpath ../third-party/flatbuffers/cmake-out/flatc) \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=$(pwd)
```

Append `-DIOS_PLATFORM=SIMULATOR` for Simulator configuration to build libraries
for `x86` architecture instead of `arm64`, which is the default.

Append `-DEXECUTORCH_BUILD_COREML_DELGATE=ON` to build CoreML backend libraries.

Append `-DEXECUTORCH_BUILD_XNNPACK=ON` to build XNNPACK backend libraries.

## Building and Copying Libraries

1. Build the libraries:

```bash
cmake --build . --config Release
```

2. Copy the libraries to the appropriate location to link against them:

Navigate to the build artifacts directory:

```bash
cd Release
```

Copy the core libraries:

```bash
mkdir -p ../../examples/apple/ios/ExecuTorchDemo/ExecuTorchDemo/Frameworks/executorch/
cp libexecutorch.a \
   libextension_data_loader.a \
   ../../examples/apple/ios/ExecuTorchDemo/ExecuTorchDemo/Frameworks/executorch/
```

For Portable CPU operators, copy additional libraries:

```bash
mkdir -p ../../examples/apple/ios/ExecuTorchDemo/ExecuTorchDemo/Frameworks/portable/
cp libportable_kernels.a \
   libportable_ops_lib.a \
   ../../examples/apple/ios/ExecuTorchDemo/ExecuTorchDemo/Frameworks/portable/
```

For CoreML delegate backend, copy additional libraries:

```bash
mkdir -p ../../examples/apple/ios/ExecuTorchDemo/ExecuTorchDemo/Frameworks/coreml/
cp libcoremldelegate.a \
   ../../examples/apple/ios/ExecuTorchDemo/ExecuTorchDemo/Frameworks/coreml/
```

For XNNPACK delegate backend, copy additional libraries:

```bash
mkdir -p ../../examples/apple/ios/ExecuTorchDemo/ExecuTorchDemo/Frameworks/xnnpack/
cp libclog.a \
   libcpuinfo.a \
   libpthreadpool.a \
   libxnnpack_backend.a \
   libXNNPACK.a \
   ../../examples/apple/ios/ExecuTorchDemo/ExecuTorchDemo/Frameworks/xnnpack/
```

Then return to the `executorch` directory:

```bash
cd ../..
```

## Model Download and Bundling

1. Download MobileNet model labels and bundle them with the app:

```bash
mkdir -p examples/apple/ios/ExecuTorchDemo/ExecuTorchDemo/Resources/Models/MobileNet/
curl https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt -o examples/apple/ios/ExecuTorchDemo/ExecuTorchDemo/Resources/Models/MobileNet/imagenet_classes.txt
```

2. Export a MobileNet v3 model backed with XNNPACK delegate and bundle it with
   the app:

```bash
export FLATC_EXECUTABLE=$(realpath third-party/flatbuffers/cmake-out/flatc)
python3 -m examples.export.export_example --model_name="mv3"
python3 -m examples.backend.xnnpack_examples --model_name="mv3" --delegate
python3 -m examples.export.coreml_export_and_delegate -m "mv3"
cp mv3.pte mv3_coreml.pte mv3_xnnpack_fp32.pte examples/apple/ios/ExecuTorchDemo/ExecuTorchDemo/Resources/Models/MobileNet/
```

## Final Steps

1. Open the project with Xcode:

```bash
open executorch/examples/apple/ios/ExecuTorchDemo/ExecuTorchDemo.xcodeproj
```

2. Set the Header Search Paths for `MobileNetClassifier` target to the directory
   containing the `executorch` folder.

3. Run the app (Cmd + R) and tests (Cmd + U).
