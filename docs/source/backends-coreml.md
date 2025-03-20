# Core ML Backend

Core ML delegate is the ExecuTorch solution to take advantage of Apple's [CoreML framework](https://developer.apple.com/documentation/coreml) for on-device ML.  With CoreML, a model can run on CPU, GPU, and the Apple Neural Engine (ANE).

## Features

- Dynamic dispatch to the CPU, GPU, and ANE.
- Supports fp32 and fp16 computation.

## Target Requirements

Below are the minimum OS requirements on various hardware for running a CoreML-delegated ExecuTorch model:
- [macOS](https://developer.apple.com/macos) >= 13.0
- [iOS](https://developer.apple.com/ios/) >= 16.0
- [iPadOS](https://developer.apple.com/ipados/) >= 16.0
- [tvOS](https://developer.apple.com/tvos/) >= 16.0

## Development Requirements
To develop you need:

- [macOS](https://developer.apple.com/macos) >= 13.0.
- [Xcode](https://developer.apple.com/documentation/xcode) >= 14.1


Before starting, make sure you install the Xcode Command Line Tools:

```bash
xcode-select --install
```

Finally you must install the CoreML backend by running the following script:
```bash
sh ./backends/apple/coreml/scripts/install_requirements.sh
```


----

## Using the CoreML Backend

To target the CoreML backend during the export and lowering process, pass an instance of the `CoreMLPartitioner` to `to_edge_transform_and_lower`. The example below demonstrates this process using the MobileNet V2 model from torchvision.

```python
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.exir import to_edge_transform_and_lower

mobilenet_v2 = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

et_program = to_edge_transform_and_lower(
    torch.export.export(mobilenet_v2, sample_inputs),
    partitioner=[CoreMLPartitioner()],
).to_executorch()

with open("mv2_coreml.pte", "wb") as file:
    et_program.write_to_file(file)
```

### Partitioner API

The CoreML partitioner API allows for configuration of the model delegation to CoreML. Passing an `CoreMLPartitioner` instance with no additional parameters will run as much of the model as possible on the CoreML backend with default settings. This is the most common use-case. For advanced use cases, the partitioner exposes the following options via the [constructor](https://github.com/pytorch/executorch/blob/14ff52ff89a89c074fc6c14d3f01683677783dcd/backends/apple/coreml/partition/coreml_partitioner.py#L60):


 - `skip_ops_for_coreml_delegation`: Allows you to skip ops for delegation by CoreML.  By default, all ops that CoreML supports will be delegated.  See [here](https://github.com/pytorch/executorch/blob/14ff52ff89a89c074fc6c14d3f01683677783dcd/backends/apple/coreml/test/test_coreml_partitioner.py#L42) for an example of skipping an op for delegation.
- `compile_specs`: A list of CompileSpec for the CoreML backend.  These control low-level details of CoreML delegation, such as the compute unit (CPU, GPU, ANE), the iOS deployment target, and the compute precision (FP16, FP32).  These are discussed more below.
- `take_over_mutable_buffer`: A boolean that indicates whether PyTorch mutable buffers in stateful models should be converted to [CoreML MLState](https://developer.apple.com/documentation/coreml/mlstate).  If set to false, mutable buffers in the PyTorch graph are converted to graph inputs and outputs to the CoreML lowered module under the hood.  Generally setting take_over_mutable_buffer to true will result in better performance, but using MLState requires iOS >= 18.0, macOS >= 15.0, and XCode >= 16.0.

#### CoreML CompileSpec

A list of CompileSpec is constructed with [CoreMLBackend.generate_compile_specs](https://github.com/pytorch/executorch/blob/14ff52ff89a89c074fc6c14d3f01683677783dcd/backends/apple/coreml/compiler/coreml_preprocess.py#L210).  Below are the available options:
- `compute_unit`: this controls the compute units (CPU, GPU, ANE) that are used by CoreML.  The default value is coremltools.ComputeUnit.ALL.  The [available options] from coremltools are:
    - coremltools.ComputeUnit.ALL (uses the CPU, GPU, and ANE)
    - coremltools.ComputeUnit.CPU_ONLY (uses the CPU only)
    - coremltools.ComputeUnit.CPU_AND_GPU (uses both the CPU and GPU, but not the ANE)
    - coremltools.ComputeUnit.CPU_AND_NE (uses both the CPU and ANE, but not the GPU)
- `minimum_deployment_target`: The minimum iOS deployment target (e.g., coremltools.target.iOS18).  The default value is coremltools.target.iOS15.
- `compute_precision`: The compute precision used by CoreML (coremltools.precision.FLOAT16, coremltools.precision.FLOAT32).  The default value is coremltools.precision.FLOAT16.  Note that the compute precision is applied no matter what dtype is specified in the exported PyTorch model.  For example, an FP32 PyTorch model will be converted to FP16 when delegating to the CoreML backend by default.  Also note that the ANE only supports FP16 precision.
- `model_type`: Whether the model should be compiled to the CoreML [mlmodelc format](https://developer.apple.com/documentation/coreml/downloading-and-compiling-a-model-on-the-user-s-device) during .pte creation ([CoreMLBackend.MODEL_TYPE.COMPILED_MODEL](https://github.com/pytorch/executorch/blob/14ff52ff89a89c074fc6c14d3f01683677783dcd/backends/apple/coreml/compiler/coreml_preprocess.py#L71)), or whether it should be compiled to mlmodelc on device ([CoreMLBackend.MODEL_TYPE.MODEL](https://github.com/pytorch/executorch/blob/14ff52ff89a89c074fc6c14d3f01683677783dcd/backends/apple/coreml/compiler/coreml_preprocess.py#L70)).  Using CoreMLBackend.MODEL_TYPE.COMPILED_MODEL and doing compilation ahead of time should improve the first time on-device model load time.


### Testing the Model

After generating the CoreML-delegated .pte, the model can be tested from Python using the ExecuTorch runtime python bindings. This can be used to sanity check the model and evaluate numerical accuracy. See [Testing the Model](using-executorch-export.md#testing-the-model) for more information.

---

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
