# Building and Running ExecuTorch with MPS Backend

In this tutorial we will walk you through the process of getting setup to build the MPS backend for ExecuTorch and running a simple model on it.

The MPS backend device maps machine learning computational graphs and primitives on the [MPS Graph](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph?language=objc) framework and tuned kernels provided by [MPS](https://developer.apple.com/documentation/metalperformanceshaders?language=objc).

::::{grid} 2
:::{grid-item-card}  What you will learn in this tutorial:
:class-card: card-prerequisites
* In this tutorial you will learn how to export [MobileNet V3](https://pytorch.org/vision/main/models/mobilenetv3.html) model to the MPS delegate.
* You will also learn how to compile and deploy the ExecuTorch runtime with the MPS delegate on macOS and iOS.
:::
:::{grid-item-card}  Tutorials we recommend you complete before this:
:class-card: card-prerequisites
* [Introduction to ExecuTorch](intro-how-it-works.md)
* [Setting up ExecuTorch](getting-started-setup.md)
* [Building ExecuTorch with CMake](runtime-build-and-cross-compilation.md)
:::
::::


## Prerequisites (Hardware and Software)

In order to be able to successfully build and run a model using the MPS backend for ExecuTorch, you'll need the following hardware and software components.

### Hardware
- [Apple Silicon](https://support.apple.com/en-us/HT211814)

### Software
- [macOS Sonoma](https://www.apple.com/macos/sonoma/) (for lowering to MPS delegate)
- [iOS 17](https://www.apple.com/ios/ios-17/) / [iPadOS 17](https://www.apple.com/ipados/ipados-17/) (for running on device)
- [Xcode 15](https://developer.apple.com/xcode/) (for building the [AOT](#aot-ahead-of-time-components) and [runtime](#runtime))

## Setting up Developer Environment

***Step 1.*** Please finish tutorial [Setting up ExecuTorch](https://pytorch.org/executorch/stable/getting-started-setup).

***Step 2.*** Install dependencies needed to lower MPS delegate:

  ```bash
  ./backends/apple/mps/install_requirements.sh
  ```

## Build

### AOT (Ahead-of-time) Components

**Compiling model for MPS delegate**:
- In this step, you will generate a simple ExecuTorch program that lowers MobileNetV3 model to the MPS delegate. You'll then pass this Program(the `.pte` file) during the runtime to run it using the MPS backend.

```bash
cd executorch
python3 -m unittest backends.apple.mps.test.test_mps --verbose -k mv3
```

### Runtime

**Building the MPS executor runner**
- In this step, you'll be building the `mps_executor_runner` that is able to run MPS lowered modules.

***Step 1***. Run the CMake build.

```bash
# Build the mps_executor_runner
rm -rf cmake-out && mkdir cmake-out && cd cmake-out && cmake -DEXECUTORCH_BUILD_MPS=1 -DEXECUTORCH_BUILD_SDK=ON -DBUCK2=/tmp/buck2 —trace .. && cmake --build . && cd ..
```

***Step 2***. Run the model using the `mps_executor_runner`.
```bash
./cmake-out/examples/apple/mps/mps_executor_runner --model_path mv3.pte --bundled_program
```

You should see the following results. Note that no output file will be generated in this example:
```
./cmake-out/examples/apple/mps/mps_executor_runner --model_path mv3.pte --bundled_program
I 00:00:00.003290 executorch:mps_executor_runner.mm:286] Model file mv3.pte is loaded.
I 00:00:00.003306 executorch:mps_executor_runner.mm:292] Program methods: 1
I 00:00:00.003308 executorch:mps_executor_runner.mm:294] Running method forward
I 00:00:00.003311 executorch:mps_executor_runner.mm:349] Setting up non-const buffer 1, size 606112.
I 00:00:00.003374 executorch:mps_executor_runner.mm:376] Setting up memory manager
I 00:00:00.003376 executorch:mps_executor_runner.mm:392] Loading method name from plan
I 00:00:00.018942 executorch:mps_executor_runner.mm:399] Method loaded.
I 00:00:00.018944 executorch:mps_executor_runner.mm:404] Loading bundled program...
I 00:00:00.018980 executorch:mps_executor_runner.mm:421] Inputs prepared.
I 00:00:00.118731 executorch:mps_executor_runner.mm:438] Model executed successfully.
I 00:00:00.122615 executorch:mps_executor_runner.mm:501] Model verified successfully.
```

## Deploying and Running on Device

***Step 1***. Create the ExecuTorch core and MPS delegate frameworks to link on iOS
```bash
cd executorch
./build/build_apple_frameworks.sh --Release --mps
```

`mps_delegate.xcframework` will be in `cmake-out` folder, along with `executorch.xcframework` and `portable_delegate.xcframework`:
```bash
cd cmake-out && ls
```

***Step 2***. Link the frameworks into your XCode project:
Go to project Target’s  `Build Phases`  -  `Link Binaries With Libraries`, click the **+** sign and add the frameworks: files located in  `Release` folder.
- `executorch.xcframework`
- `portable_delegate.xcframework`
- `mps_delegate.xcframework`

From the same page, include the needed libraries for the MPS delegate:
- `MetalPerformanceShaders.framework`
- `MetalPerformanceShadersGraph.framework`
- `Metal.framework`

In this tutorial, you have learned how to lower a model to the MPS delegate, build the mps_executor_runner and run a lowered model through the MPS delegate, or directly on device using the MPS delegate static library.


## Frequently encountered errors and resolution.

If you encountered any bugs or issues following this tutorial please file a bug/issue on the ExecuTorch repository, with hashtag **#mps**.
