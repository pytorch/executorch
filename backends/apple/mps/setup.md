# Building and Running ExecuTorch on MPS Backend

In this tutorial we will walk you through the process of getting setup to build the MPS backend for ExecuTorch and running a simple model on it.

The MPS backend device maps machine learning computational graphs and primitives on the [MPS Graph](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph?language=objc) framework and tuned kernels provided by [MPS](https://developer.apple.com/documentation/metalperformanceshaders?language=objc).

## Prerequisites (Hardware and Software)

In order to be able to successfully build and run a model using the MPS backend for ExecuTorch, you'll need the following hardware and software components.

### Hardware
- [Apple Silicon](https://support.apple.com/en-us/HT211814)

### Software
- [macOS Sonoma](https://www.apple.com/macos/sonoma/) (for lowering to MPS delegate)
- [iOS 17](https://www.apple.com/ios/ios-17/) / [iPadOS 17](https://www.apple.com/ipados/ipados-17/) (for running on device)


## Setting up Developer Environment

***Step 1.*** Please finish tutorial [Setting up executorch](../../../docs/website/docs/tutorials/00_setting_up_executorch.md).

***Step 2.*** Install dependencies needed to lower MPS delegate:

  ```bash
  bash ./backends/apple/mps/install_requirements.sh
  ```

## Build

### AOT (Ahead-of-time) Components

**Compiling model for MPS delegate**:
- In this step, you will generate a simple ExecuTorch program that lowers MobileNetV3 model to the MPS delegate. You'll then pass this Program(the `.pte` file) during the runtime to run it using the MPS backend.

```bash
cd executorch
python3 -m unittest backends.apple.mps.test.test_mps --verbose -k test_mps_backend_pixel_shuffle
```

### Runtime

**Building the MPS executor runner**
- In this step, you'll be building the `mps_executor_runner` that is able to run MPS lowered modules.

***Step 1***. Run the CMake build.
```bash
# Build the mps_executor_runner
rm -rf cmake-out && mkdir cmake-out && cd cmake-out && cmake -DEXECUTORCH_BUILD_MPS=1 -DBUCK2=/tmp/buck2 —trace .. && cmake --build .
```

***Step 2***. Run the model using the `mps_executor_runner`.
```bash
./examples/apple/mps/mps_executor_runner --model_path mps_backend_pixel_shuffle.pte --bundled_program
```

You should see the following results. Note that no output file will be generated in this example:
```
./examples/apple/mps/mps_executor_runner --model_path mps_backend_pixel_shuffle.pte --bundled_program
I 00:00:00.000452 executorch:mps_executor_runner.mm:286] Model file mps_backend_pixel_shuffle.pte is loaded.
I 00:00:00.000467 executorch:mps_executor_runner.mm:292] Program methods: 1
I 00:00:00.000469 executorch:mps_executor_runner.mm:294] Running method forward
I 00:00:00.000487 executorch:mps_executor_runner.mm:349] Setting up non-const buffer 1, size 5376.
I 00:00:00.000493 executorch:mps_executor_runner.mm:376] Setting up memory manager
I 00:00:00.000495 executorch:mps_executor_runner.mm:392] Loading method name from plan
I 00:00:00.008456 executorch:mps_executor_runner.mm:399] Method loaded.
I 00:00:00.008458 executorch:mps_executor_runner.mm:404] Loading bundled program...
I 00:00:00.061173 executorch:mps_executor_runner.mm:501] Model verified successfully.
```

## Deploying and Running on Device

***Step 1***. Configure the libraries for iOS:

```bash
cd executorch
rm -rf cmake-out && mkdir cmake-out && cd cmake-out
cmake .. -G Xcode \
    -DCMAKE_TOOLCHAIN_FILE=../third-party/pytorch/cmake/iOS.cmake \
    -DBUCK2=/tmp/buck2 \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DFLATC_EXECUTABLE=$(realpath ../third-party/flatbuffers/cmake-out/flatc) \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=$(pwd) \
    -DEXECUTORCH_BUILD_MPS=ON
```

***Step 2***. Generate the ExecuTorch core libraries and MPS delegate static library to link on iOS:
```bash
cmake --build . --config Release
```

`libmpsdelegate.a` will be in Release folder:
```bash
cd Release
```

***Step 3***. Link the core libraries into your XCode project:

Go to project Target’s  `Build Phases`  -  `Link Binaries With Libraries`, click the **+** sign and add all the core libraries: files located in  `Release` folder.
- `libexecutorch.a`
- `libextension_data_loader.a`
- `libmpsdelegate.a`

From the same page, include the needed libraries for the MPS delegate:
- `MetalPerformanceShaders.framework`
- `MetalPerformanceShadersGraph.framework`
- `Metal.framework`

In this tutorial, you have learned how to lower a model to the MPS delegate, build the mps_executor_runner and run a lowered model through the MPS delegate, or directly on device using the MPS delegate static library.
