# Building and Running ExecuTorch with Core ML Backend

Core ML delegate uses Core ML APIs to enable running neural networks via Apple's hardware acceleration. For more about Core ML you can read [here](https://developer.apple.com/documentation/coreml). In this tutorial, we will walk through the steps of lowering a PyTorch model to Core ML delegate


::::{grid} 2
:::{grid-item-card}  What you will learn in this tutorial:
:class-card: card-prerequisites
* In this tutorial you will learn how to export [MobileNet V3](https://pytorch.org/vision/main/models/mobilenetv3.html) model so that it runs on Core ML backend.
* You will also learn how to deploy and run the exported model on a supported Apple device.
:::
:::{grid-item-card}  Tutorials we recommend you complete before this:
:class-card: card-prerequisites
* [Introduction to ExecuTorch](intro-how-it-works.md)
* [Setting up ExecuTorch](getting-started-setup.md)
* [Building ExecuTorch with CMake](runtime-build-and-cross-compilation.md)
* [ExecuTorch iOS Demo App](demo-apps-ios.md)
:::
::::


## Prerequisites (Hardware and Software)

In order to be able to successfully build and run the ExecuTorch's Core ML backend you'll need the following hardware and software components.

### Hardware:
- A [mac](https://www.apple.com/mac/) system for building.
- A [mac](https://www.apple.com/mac/) or [iPhone](https://www.apple.com/iphone/) or [iPad](https://www.apple.com/ipad/) or [Apple TV](https://www.apple.com/tv-home/) device for running the model.

### Software:

- [Xcode](https://developer.apple.com/documentation/xcode) >= 14.1, [macOS](https://developer.apple.com/macos) >= 13.0 for building.
- [macOS](https://developer.apple.com/macos) >= 13.0, [iOS](https://developer.apple.com/ios/) >= 16.0, [iPadOS](https://developer.apple.com/ipados/) >= 16.0, and [tvOS](https://developer.apple.com/tvos/) >= 16.0 for running the model.

## Setting up your developer environment

1. Make sure that you have completed the ExecuTorch setup tutorials linked to at the top of this page and setup the environment.
2. Run `install_requirements.sh` to install dependencies required by the **Core ML** backend.

```bash
cd executorch
./backends/apple/coreml/scripts/install_requirements.sh
```
3. Install [Xcode](https://developer.apple.com/xcode/).
4. Install Xcode Command Line Tools.

```bash
xcode-select --install
```

## Build

### AOT (Ahead-of-time) components:


**Exporting a Core ML delegated Program**:
- In this step, you will lower the [MobileNet V3](https://pytorch.org/vision/main/models/mobilenetv3.html) model to the Core ML backend and export the ExecuTorch program. You'll then deploy and run the exported program on a supported Apple device using Core ML backend.
```bash
cd executorch

# Generates ./mv3_coreml_all.pte file.
python3 -m examples.apple.coreml.scripts.export --model_name mv3
```

- Core ML backend uses [coremltools](https://apple.github.io/coremltools/docs-guides/source/overview-coremltools.html) to lower [Edge dialect](ir-exir.md#edge-dialect) to Core ML format and then bundles it in the `.pte` file.


### Runtime:

**Running a Core ML delegated Program**:
1. Build the runner.
```bash
cd executorch

# Builds `coreml_executor_runner`.
./examples/apple/coreml/scripts/build_executor_runner.sh
```
2. Run the CoreML delegated program.
```bash
cd executorch

# Runs the exported mv3 model using the Core ML backend.
./coreml_executor_runner --model_path mv3_coreml_all.pte
```

**Profiling a Core ML delegated Program**:

Note that profiling is supported on [macOS](https://developer.apple.com/macos) >= 14.4.

1. [Optional] Generate an [ETRecord](./etrecord.rst) when exporting your model.
```bash
cd executorch

# Generates `mv3_coreml_all.pte` and `mv3_coreml_etrecord.bin` files.
python3 -m examples.apple.coreml.scripts.export --model_name mv3 --generate_etrecord
```

2. Build the runner.
```bash
# Builds `coreml_executor_runner`.
./examples/apple/coreml/scripts/build_executor_runner.sh
```
3. Run and generate an [ETDump](./etdump.md).
```bash
cd executorch

# Generate the ETDump file.
./coreml_executor_runner --model_path mv3_coreml_all.pte --profile_model --etdump_path etdump.etdp
```

4. Create an instance of the [Inspector API](./model-inspector.rst) by passing in the [ETDump](./etdump.md) you have sourced from the runtime along with the optionally generated [ETRecord](./etrecord.rst) from step 1 or execute the following command in your terminal to display the profiling data table.
```bash
python examples/apple/coreml/scripts/inspector_cli.py --etdump_path etdump.etdp --etrecord_path mv3_coreml.bin
```


## Deploying and running on a device

**Running the Core ML delegated Program in the Demo iOS App**:
1. Please follow the [Export Model](demo-apps-ios.md#models-and-labels) step of the tutorial to bundle the exported [MobileNet V3](https://pytorch.org/vision/main/models/mobilenetv3.html) program. You only need to do the Core ML part.

2. Complete the [Build Runtime and Backends](demo-apps-ios.md#build-runtime-and-backends) section of the tutorial. When building the frameworks you only need the `coreml` option.

3. Complete the [Final Steps](demo-apps-ios.md#final-steps) section of the tutorial to build and run the demo app.

<br>**Running the Core ML delegated Program in your App**
1. Build frameworks, running the following will create a `executorch.xcframework` and `coreml_backend.xcframework` in the `cmake-out` directory.
```bash
cd executorch
./build/build_apple_frameworks.sh --coreml
```
2. Create a new [Xcode project](https://developer.apple.com/documentation/xcode/creating-an-xcode-project-for-an-app#) or open an existing project.

3. Drag the `executorch.xcframework` and `coreml_backend.xcframework` generated from Step 2 to Frameworks.

4. Go to the project's [Build Phases](https://developer.apple.com/documentation/xcode/customizing-the-build-phases-of-a-target) -  Link Binaries With Libraries, click the + sign, and add the following frameworks:
```
executorch.xcframework
coreml_backend.xcframework
Accelerate.framework
CoreML.framework
libsqlite3.tbd
```
5. Add the exported program to the [Copy Bundle Phase](https://developer.apple.com/documentation/xcode/customizing-the-build-phases-of-a-target#Copy-files-to-the-finished-product) of your Xcode target.

6. Please follow the [Runtime APIs Tutorial](extension-module.md) to integrate the code for loading an ExecuTorch program.

7. Update the code to load the program from the Application's bundle.
``` objective-c
NSURL *model_url = [NBundle.mainBundle URLForResource:@"mv3_coreml_all" extension:@"pte"];

Result<executorch::extension::FileDataLoader> loader =
    executorch::extension::FileDataLoader::from(model_url.path.UTF8String);
```

8. Use [Xcode](https://developer.apple.com/documentation/xcode/building-and-running-an-app#Build-run-and-debug-your-app) to deploy the application on the device.

9. The application can now run the [MobileNet V3](https://pytorch.org/vision/main/models/mobilenetv3.html) model on the Core ML backend.

<br>In this tutorial, you have learned how to lower the [MobileNet V3](https://pytorch.org/vision/main/models/mobilenetv3.html) model to the Core ML backend, deploy, and run it on an Apple device.

## Frequently encountered errors and resolution.

If you encountered any bugs or issues following this tutorial please file a bug/issue [here](https://github.com/pytorch/executorch/issues) with tag #coreml.
