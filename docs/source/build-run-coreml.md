<!---- To make this progress bar work users will need to modify source/_templates/layout.html >
<!---- To make this page show up in the tutorials section users will need to add an entry in source/index.rst under the Tutorials section>

<!---- DO NOT MODIFY Progress Bar Start --->

<div class="progress-bar-wrapper">
   <div class="progress-bar-item">
     <div class="step-number" id="step-1">1</div>
     <span class="step-caption" id="caption-1"></span>
   </div>
   <div class="progress-bar-item">
     <div class="step-number" id="step-2">2</div>
     <span class="step-caption" id="caption-2"></span>
   </div>
   <div class="progress-bar-item">
     <div class="step-number" id="step-3">3</div>
     <span class="step-caption" id="caption-3"></span>
   </div>
   <div class="progress-bar-item">
     <div class="step-number" id="step-4">4</div>
     <span class="step-caption" id="caption-4"></span>
   </div>
</div>

<!---- DO NOT MODIFY Progress Bar End--->

# Building and Running ExecuTorch on CoreML Backend

In this tutorial we will walk you through all the steps required to export, deploy, and run a model on CoreML backend.

CoreML backend uses the [CoreML framework](https://developer.apple.com/documentation/coreml) to run the model. CoreML framework optimizes on-device performance by leveraging the CPU, GPU, and Neural Engine while minimizing its memory footprint and power consumption.  


::::{grid} 2
:::{grid-item-card}  What you will learn in this tutorial:
:class-card: card-learn
* In this tutorial you will learn how to export [MobileNet V3](https://pytorch.org/vision/main/models/mobilenetv3.html) model so that it runs on CoreML backend. 
* You will also learn how to deploy and run the exported model on a supported Apple device.
:::
:::{grid-item-card}  Tutorials we recommend you complete before this:
:class-card: card-prerequisites
* [Introduction to ExecuTorch](intro-how-it-works.md)
* [Setting up ExecuTorch](getting-started-setup.md)
* [Building ExecuTorch with CMake](runtime-build-and-cross-compilation.md)
:::
::::


## Prerequisites (Hardware and Software)

In order to be able to successfully build and run the ExecuTorch's CoreML backend you'll need the following hardware and software components.

### Hardware:
- A [mac](https://www.apple.com/mac/]) system for building.
- A [mac](https://www.apple.com/mac/]),[iPhone](https://www.apple.com/iphone/) or [iPad](https://www.apple.com/ipad/) or [Apple TV](https://www.apple.com/tv-home/) device for running the model.

### Software:

- [Xcode](https://developer.apple.com/documentation/xcode) >= 14.1, [macOS](https://developer.apple.com/macos) >= 13.0 for building.
- [macOS](https://developer.apple.com/macos) >= 13.0, [iOS](https://developer.apple.com/ios/) >= 16.0, [iPadOS](https://developer.apple.com/ipados/) >= 16.0, and [tvOS](https://developer.apple.com/tvos/) >= 16.0 for running the model. 

## Setting up your developer environment

1. Make sure that you have completed the ExecuTorch setup tutorials linked to at the top of this page and setup the environment.
2. Run `install_requirements.sh` to install dependencies required by the **CoreML** backend.

```bash
cd executorch
sh backends/apple/coreml/scripts/install_requirements.sh   
```
3. Install [Xcode](https://developer.apple.com/xcode/).
4. Install Xcode Command Line Tools.

```bash
xcode-select --install
```

## Build

### AOT (Ahead-of-time) components:


**Exporting a CoreML delegated Program**:
- In this step, you will lower the [MobileNet V3](https://pytorch.org/vision/main/models/mobilenetv3.html) model to the CoreML backend and export the ExecuTorch program. You'll then deploy and run the exported program on a supported Apple device using CoreML backend. 
```bash
cd executorch

# Generates ./mv3_coreml_all.pte file.
python3 -m examples.apple.coreml.scripts.export_and_delegate --model_name mv3 
```

- CoreML backend uses [coremltools](https://apple.github.io/coremltools/docs-guides/source/overview-coremltools.html) to lower [Edge dialect](ir-exir.md#edge-dialect) to CoreML format and then bundles it in the `.pte` file.


### Runtime:

**Running the CoreML delegated Program**:
1. Build the runner.
```bash
cd executorch

# Generates ./coreml_executor_runner.
sh examples/apple/coreml/scripts/build_executor_runner.sh
```
2. Run the exported program.
```bash
cd executorch

# Runs the exported mv3 model on the CoreML backend.
./coreml_executor_runner --model_path mv3_coreml_all.pte
```

## Deploying and running on a device

1. Build **CoreML** delegate. The following will create a `executorch.xcframework` in the `cmake-out` directory.
```bash
cd executorch
./build/build_apple_frameworks.sh --Release --coreml
```
2. Open the project in Xcode, and drag the `executorch.xcframework` generated from Step 2 to Frameworks.

3. Go to project Target’s Build Phases -  Link Binaries With Libraries, click the + sign, and add the following frameworks:
```
- executorch.xcframework
- coreml_backend.xcframework
- Accelerate.framework
- CoreML.framework
- libsqlite3.tbd
```
4. Add the exported program to the [Copy Bundle Phase](https://developer.apple.com/documentation/xcode/customizing-the-build-phases-of-a-target#Copy-files-to-the-finished-product) of your Xcode target.
 
5. Please follow the [running a model](running-a-model-cpp-tutorial.md) tutorial to integrate the code for loading a ExecuTorch program.

6. Update the code to load the program from the Application's bundle.
``` cpp
using namespace torch::executor;

NSURL *model_url = [NBundle.mainBundle URLForResource:@"mv3_coreml_all" extension:@"pte"];

Result<util::FileDataLoader> loader =
        util::FileDataLoader::from(model_url.path.UTF8String);

```

7. Use [Xcode](https://developer.apple.com/documentation/xcode/building-and-running-an-app#Build-run-and-debug-your-app) to deploy the application on the device. 

8. The application can now run the [MobileNet V3](https://pytorch.org/vision/main/models/mobilenetv3.html) model on the CoreML backend.  

<br>In this tutorial, you have learned how to lower the [MobileNet V3](https://pytorch.org/vision/main/models/mobilenetv3.html) model to the CoreML backend, deploy, and run it on an Apple device.

## Frequently encountered errors and resolution.

If you encountered any bugs or issues following this tutorial please file a bug/issue [here](https://github.com/pytorch/executorch/issues) with tag #coreml.
