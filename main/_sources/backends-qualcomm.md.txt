# Qualcomm AI Engine Backend

In this tutorial we will walk you through the process of getting started to
build ExecuTorch for Qualcomm AI Engine Direct and running a model on it.

Qualcomm AI Engine Direct is also referred to as QNN in the source and documentation.


<!----This will show a grid card on the page----->
::::{grid} 2
:::{grid-item-card}  What you will learn in this tutorial:
:class-card: card-prerequisites
* In this tutorial you will learn how to lower and deploy a model for Qualcomm AI Engine Direct.
:::
:::{grid-item-card}  Tutorials we recommend you complete before this:
:class-card: card-prerequisites
* [Introduction to ExecuTorch](intro-how-it-works.md)
* [Getting Started](getting-started.md)
* [Building ExecuTorch with CMake](using-executorch-building-from-source.md)
:::
::::


## What's Qualcomm AI Engine Direct?

[Qualcomm AI Engine Direct](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk)
is designed to provide unified, low-level APIs for AI development.

Developers can interact with various accelerators on Qualcomm SoCs with these set of APIs, including
Kryo CPU, Adreno GPU, and Hexagon processors. More details can be found [here](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/overview.html).

Currently, this ExecuTorch Backend can delegate AI computations to Hexagon processors through Qualcomm AI Engine Direct APIs.


## Prerequisites (Hardware and Software)

### Host OS

The QNN Backend is currently verified on the following Linux host operating systems:

- **Ubuntu 22.04 LTS (x64)**
- **CentOS Stream 9**
- **Windows Subsystem for Linux (WSL)** with Ubuntu 22.04

In general, we verify the backend on the same OS versions that the QNN SDK is officially validated against.  
The exact supported versions are documented in the QNN SDK.

#### Windows (WSL) Setup

To install Ubuntu 22.04 on WSL, run the following command in PowerShell or Windows Terminal:

```bash
wsl --install -d ubuntu 22.04
```

This command will install WSL and set up Ubuntu 22.04 as the default Linux distribution.

For more details and troubleshooting, refer to the official Microsoft WSL installation guide:
👉 [Install WSL | Microsoft Learn](https://learn.microsoft.com/en-us/windows/wsl/install)

### Hardware:
You will need an Android / Linux device with adb-connected running on one of below Qualcomm SoCs:
 - SA8295
 - SM8450 (Snapdragon 8 Gen 1)
 - SM8475 (Snapdragon 8 Gen 1+)
 - SM8550 (Snapdragon 8 Gen 2)
 - SM8650 (Snapdragon 8 Gen 3)
 - SM8750 (Snapdragon 8 Elite)
 - SSG2115P
 - SSG2125P
 - SXR1230P (Linux Embedded)
 - SXR2230P
 - SXR2330P

This example is verified with SM8550 and SM8450.

### Software:

 - Follow ExecuTorch recommended Python version.
 - A compiler to compile AOT parts, e.g., the GCC compiler comes with Ubuntu LTS. g++ version need to be 13 or higher.
 - [Android NDK](https://developer.android.com/ndk). This example is verified with NDK 26c.
 - (Optional) Target toolchain for linux embedded platform.
 - [Qualcomm AI Engine Direct SDK](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk)
   - Click the "Get Software" button to download the latest version of the QNN SDK.
   - Although newer versions are available, we have verified and recommend using QNN 2.37.0 for stability.
   - You can download it directly from the following link: [QNN 2.37.0](https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.37.0.250724/v2.37.0.250724.zip)

The directory with installed Qualcomm AI Engine Direct SDK looks like:
```
├── benchmarks
├── bin
├── docs
├── examples
├── include
├── lib
├── LICENSE.pdf
├── NOTICE.txt
├── NOTICE_WINDOWS.txt
├── QNN_NOTICE.txt
├── QNN_README.txt
├── QNN_ReleaseNotes.txt
├── ReleaseNotes.txt
├── ReleaseNotesWindows.txt
├── sdk.yaml
└── share
```


## Setting up your developer environment

### Conventions

`$QNN_SDK_ROOT` refers to the root of Qualcomm AI Engine Direct SDK,
i.e., the directory containing `QNN_README.txt`.

`$ANDROID_NDK_ROOT` refers to the root of Android NDK.

`$EXECUTORCH_ROOT` refers to the root of executorch git repository.

### Setup environment variables

We set `LD_LIBRARY_PATH` to make sure the dynamic linker can find QNN libraries.

Further, we set `PYTHONPATH` because it's easier to develop and import ExecuTorch
Python APIs.

```bash
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang/:$LD_LIBRARY_PATH
export PYTHONPATH=$EXECUTORCH_ROOT/..
```

## Build

An example script for the below building instructions is [here](https://github.com/pytorch/executorch/blob/main/backends/qualcomm/scripts/build.sh).
We recommend to use the script because the ExecuTorch build-command can change from time to time.
The above script is actively used. It is updated more frequently than this tutorial.
An example usage is
```bash
cd $EXECUTORCH_ROOT
# android target
./backends/qualcomm/scripts/build.sh
# (optional) linux embedded target
./backends/qualcomm/scripts/build.sh --enable_linux_embedded
# for release build
./backends/qualcomm/scripts/build.sh --release
```


## Deploying and running on device

### AOT compile a model

Refer to [this script](https://github.com/pytorch/executorch/blob/main/examples/qualcomm/scripts/deeplab_v3.py) for the exact flow.
We use deeplab-v3-resnet101 as an example in this tutorial. Run below commands to compile:

```bash
cd $EXECUTORCH_ROOT

python -m examples.qualcomm.scripts.deeplab_v3 -b build-android -m SM8550 --compile_only --download
```

You might see something like below:

```
[INFO][Qnn ExecuTorch] Destroy Qnn context
[INFO][Qnn ExecuTorch] Destroy Qnn device
[INFO][Qnn ExecuTorch] Destroy Qnn backend

opcode         name                      target                       args                           kwargs
-------------  ------------------------  ---------------------------  -----------------------------  --------
placeholder    arg684_1                  arg684_1                     ()                             {}
get_attr       lowered_module_0          lowered_module_0             ()                             {}
call_function  executorch_call_delegate  executorch_call_delegate     (lowered_module_0, arg684_1)   {}
call_function  getitem                   <built-in function getitem>  (executorch_call_delegate, 0)  {}
call_function  getitem_1                 <built-in function getitem>  (executorch_call_delegate, 1)  {}
output         output                    output                       ([getitem_1, getitem],)        {}
```

The compiled model is `./deeplab_v3/dlv3_qnn.pte`.


### Test model inference on QNN HTP emulator

We can test model inferences before deploying it to a device by HTP emulator.

Let's build `qnn_executor_runner` for a x64 host:
```bash
# assuming the AOT component is built.
cd $EXECUTORCH_ROOT/build-x86
cmake ../examples/qualcomm \
  -DCMAKE_PREFIX_PATH="$PWD/lib/cmake/ExecuTorch;$PWD/third-party/gflags;" \
  -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
  -DPYTHON_EXECUTABLE=python3 \
  -Bexamples/qualcomm

cmake --build examples/qualcomm -j$(nproc)

# qnn_executor_runner can be found under examples/qualcomm/executor_runner
# The full path is $EXECUTORCH_ROOT/build-x86/examples/qualcomm/executor_runner/qnn_executor_runner
ls examples/qualcomm/executor_runner
```

To run the HTP emulator, the dynamic linker needs to access QNN libraries and `libqnn_executorch_backend.so`.
We set the below two paths to `LD_LIBRARY_PATH` environment variable:
  1. `$QNN_SDK_ROOT/lib/x86_64-linux-clang/`
  2. `$EXECUTORCH_ROOT/build-x86/lib/`

The first path is for QNN libraries including HTP emulator. It has been configured in the AOT compilation section.

The second path is for `libqnn_executorch_backend.so`.

So, we can run `./deeplab_v3/dlv3_qnn.pte` by:
```bash
cd $EXECUTORCH_ROOT/build-x86
export LD_LIBRARY_PATH=$EXECUTORCH_ROOT/build-x86/lib/:$LD_LIBRARY_PATH
examples/qualcomm/executor_runner/qnn_executor_runner --model_path ../deeplab_v3/dlv3_qnn.pte
```

We should see some outputs like the below. Note that the emulator can take some time to finish.
```bash
I 00:00:00.354662 executorch:qnn_executor_runner.cpp:213] Method loaded.
I 00:00:00.356460 executorch:qnn_executor_runner.cpp:261] ignoring error from set_output_data_ptr(): 0x2
I 00:00:00.357991 executorch:qnn_executor_runner.cpp:261] ignoring error from set_output_data_ptr(): 0x2
I 00:00:00.357996 executorch:qnn_executor_runner.cpp:265] Inputs prepared.

I 00:01:09.328144 executorch:qnn_executor_runner.cpp:414] Model executed successfully.
I 00:01:09.328159 executorch:qnn_executor_runner.cpp:421] Write etdump to etdump.etdp, Size = 424
[INFO] [Qnn ExecuTorch]: Destroy Qnn backend parameters
[INFO] [Qnn ExecuTorch]: Destroy Qnn context
[INFO] [Qnn ExecuTorch]: Destroy Qnn device
[INFO] [Qnn ExecuTorch]: Destroy Qnn backend
```

### Run model inference on an Android smartphone with Qualcomm SoCs

***Step 1***. We need to push required QNN libraries to the device.

```bash
# make sure you have write-permission on below path.
DEVICE_DIR=/data/local/tmp/executorch_qualcomm_tutorial/
adb shell "mkdir -p ${DEVICE_DIR}"
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnSystem.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV69Stub.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV73Stub.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV75Stub.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV79Stub.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/hexagon-v69/unsigned/libQnnHtpV69Skel.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so ${DEVICE_DIR}
```

***Step 2***.  We also need to indicate dynamic linkers on Android and Hexagon
where to find these libraries by setting `ADSP_LIBRARY_PATH` and `LD_LIBRARY_PATH`.
So, we can run `qnn_executor_runner` like

```bash
adb push ./deeplab_v3/dlv3_qnn.pte ${DEVICE_DIR}
adb push ${EXECUTORCH_ROOT}/build-android/examples/qualcomm/executor_runner/qnn_executor_runner ${DEVICE_DIR}
adb push ${EXECUTORCH_ROOT}/build-android/lib/libqnn_executorch_backend.so ${DEVICE_DIR}
adb shell "cd ${DEVICE_DIR} \
           && export LD_LIBRARY_PATH=${DEVICE_DIR} \
           && export ADSP_LIBRARY_PATH=${DEVICE_DIR} \
           && ./qnn_executor_runner --model_path ./dlv3_qnn.pte"
```

You should see something like below:

```
I 00:00:00.257354 executorch:qnn_executor_runner.cpp:213] Method loaded.
I 00:00:00.323502 executorch:qnn_executor_runner.cpp:262] ignoring error from set_output_data_ptr(): 0x2
I 00:00:00.357496 executorch:qnn_executor_runner.cpp:262] ignoring error from set_output_data_ptr(): 0x2
I 00:00:00.357555 executorch:qnn_executor_runner.cpp:265] Inputs prepared.
I 00:00:00.364824 executorch:qnn_executor_runner.cpp:414] Model executed successfully.
I 00:00:00.364875 executorch:qnn_executor_runner.cpp:425] Write etdump to etdump.etdp, Size = 424
[INFO] [Qnn ExecuTorch]: Destroy Qnn backend parameters
[INFO] [Qnn ExecuTorch]: Destroy Qnn context
[INFO] [Qnn ExecuTorch]: Destroy Qnn backend
```

The model is merely executed. If we want to feed real inputs and get model outputs, we can use
```bash
cd $EXECUTORCH_ROOT
# android
python -m examples.qualcomm.scripts.deeplab_v3 -b build-android -m SM8550 --download -s <device_serial>
# (optional) linux embedded
python -m examples.qualcomm.scripts.deeplab_v3 -b build-oe-linux -m SXR1230P --download -s <device_serial> -t aarch64-oe-linux-gcc-9.3
```
The `<device_serial>` can be found by `adb devices` command.

After the above command, pre-processed inputs and outputs are put in `$EXECUTORCH_ROOT/deeplab_v3` and `$EXECUTORCH_ROOT/deeplab_v3/outputs` folder.

The command-line arguments are written in [utils.py](https://github.com/pytorch/executorch/blob/main/examples/qualcomm/utils.py#L139).
The model, inputs, and output location are passed to `qnn_executorch_runner` by `--model_path`, `--input_list_path`, and `--output_folder_path`.

### Run [Android LlamaDemo](https://github.com/meta-pytorch/executorch-examples/tree/main/llm/android/LlamaDemo) with QNN backend

`$DEMO_APP` refers to the root of the executorch android demo, i.e., the directory containing `build.gradle.kts`.

***Step 1***: Rebuild ExecuTorch AAR

```bash
# Build the AAR
cd $EXECUTORCH_ROOT
export BUILD_AAR_DIR=$EXECUTORCH_ROOT/aar-out
./scripts/build_android_library.sh
```

***Step 2***: Copy AAR to Android Project

```bash
cp $EXECUTORCH_ROOT/aar-out/executorch.aar \
   $DEMO_APP/app/libs/executorch.aar
```

***Step 3***: Build Android APK

```bash
cd $DEMO_APP
./gradlew clean assembleDebug -PuseLocalAar=true
```

***Step 4***: Install on Device

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

***Step 5***: Push model

```bash
adb shell mkdir -p /data/local/tmp/llama
adb push model.pte /data/local/tmp/llama
adb push tokenizer.bin /data/local/tmp/llama
```

***Step 6***: Run the Llama Demo

- Open the App on Android
- Select `QUALCOMM` backend
- Select `model.pte` Model
- Select `tokenizer.bin` Tokenizer
- Select Model Type
- Click LOAD MODEL
- It should show `Successfully loaded model.`


#### Verification Steps

***Step 1***. Verify AAR Contains Your Changes

```bash
# Check for debug strings in the AAR
unzip -p $DEMO_APP/app/libs/executorch.aar jni/arm64-v8a/libexecutorch.so | \
  strings | grep "QNN"   # Replace "QNN" with your actual debug string if needed
```

If found, your changes are in the AAR.

***Step 2***. Verify APK Contains Correct Libraries

```bash
# Check QNN library version in APK
cd $DEMO_APP
unzip -l app/build/outputs/apk/debug/app-debug.apk | grep "libQnnHtp.so"
```

Expected size for QNN 2.37.0: ~2,465,440 bytes

***Step 3***. Monitor Logs During Model Loading

```bash
adb logcat -c
adb logcat | grep -E "ExecuTorch"
```

#### Common Issues and Solutions

##### Issue 1: Error 18 (InvalidArgument)

- **Cause**: Wrong parameter order in Runner constructor or missing QNN config

- **Solution**: Check `$EXECUTORCH_ROOT/examples/qualcomm/oss_scripts/llama/runner/runner.h` for the correct constructor signature.

##### Issue 2: Error 1 (Internal) with QNN API Version Mismatch

- **Symptoms**:

    ```
    W [Qnn ExecuTorch]: Qnn API version 2.33.0 is mismatched
    E [Qnn ExecuTorch]: Using newer context binary on old SDK
    E [Qnn ExecuTorch]: Can't create context from binary. Error 5000
    ```

- **Cause**: Model compiled with QNN SDK version X but APK uses QNN runtime version Y

- **Solution**:
    - Update `build.gradle.kts` with matching QNN runtime version

    > **Note:** The version numbers below (`2.33.0` and `2.37.0`) are examples only. Please check for the latest compatible QNN runtime version or match your QNN SDK version to avoid API mismatches.

    **Before**:
    ```kotlin
    implementation("com.qualcomm.qti:qnn-runtime:2.33.0")
    ```
    
    **After**:
    ```kotlin
    implementation("com.qualcomm.qti:qnn-runtime:2.37.0")
    ```

    - Or recompile model with matching QNN SDK version

##### Issue 3: Native Code Changes Not Applied

- **Symptoms**:
    - Debug logs don't appear
    - Behavior doesn't change

- **Cause**:
    - Gradle using Maven dependency instead of local AAR

- **Solution**:
    - Always build with `-PuseLocalAar=true` flag

##### Issue 4: Logs Not Appearing

- **Cause**: Wrong logging tag filter

- **Solution**: QNN uses "ExecuTorch" tag:

    ```bash
    adb logcat | grep "ExecuTorch"
    ```

## Supported model list

Please refer to `$EXECUTORCH_ROOT/examples/qualcomm/scripts/` and `$EXECUTORCH_ROOT/examples/qualcomm/oss_scripts/` to the list of supported models.

Each script demonstrates:
- Model export (torch.export)
- Quantization (PTQ/QAT)
- Lowering and compilation to QNN delegate

Deployment on device or HTP emulator

## How to Support a Custom Model in HTP Backend

### Step-by-Step Implementation Guide

Please reference [the simple example](https://github.com/pytorch/executorch/blob/main/examples/qualcomm/scripts/export_example.py) and [more complicated examples](https://github.com/pytorch/executorch/tree/main/examples/qualcomm/scripts) for reference
#### Step 1: Prepare Your Model
```python
import torch

# Initialize your custom model
model = YourModelClass().eval()  # Your custom PyTorch model

# Create example inputs (adjust shape as needed)
example_inputs = (torch.randn(1, 3, 224, 224),)  # Example input tensor
```

#### Step 2: [Optional] Quantize Your Model
Choose between quantization approaches, post training quantization (PTQ) or quantization aware training (QAT):
```python
from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, prepare_qat_pt2e, convert_pt2e

quantizer = QnnQuantizer()
m = torch.export.export(model, example_inputs, strict=True).module()

# PTQ (Post-Training Quantization)
if quantization_type == "ptq":
    prepared_model = prepare_pt2e(m, quantizer)
    # Calibration loop would go here
    prepared_model(*example_inputs)

# QAT (Quantization-Aware Training)
elif quantization_type == "qat":
    prepared_model = prepare_qat_pt2e(m, quantizer)
    # Training loop would go here
    for _ in range(training_steps):
        prepared_model(*example_inputs)

# Convert to quantized model
quantized_model = convert_pt2e(prepared_model)
```

The `QNNQuantizer` is configurable, with the default setting being **8a8w**. For advanced users, refer to the [`QnnQuantizer`](https://github.com/pytorch/executorch/blob/main/backends/qualcomm/quantizer/quantizer.py) documentation for details.

##### Supported Quantization Schemes
- **8a8w** (default)
- **16a16w**
- **16a8w**
- **16a4w**
- **16a4w_block**

##### Customization Options
- **Per-node annotation**: Use `custom_quant_annotations`.
- **Per-module (`nn.Module`) annotation**: Use `submodule_qconfig_list`.

##### Additional Features
- **Node exclusion**: Discard specific nodes via `discard_nodes`.
- **Blockwise quantization**: Configure block sizes with `block_size_map`.


For practical examples, see [`test_qnn_delegate.py`](https://github.com/pytorch/executorch/blob/main/backends/qualcomm/tests/test_qnn_delegate.py).


#### Step 3: Configure Compile Specs
During this step, you will need to specify the target SoC, data type, and other QNN compiler spec.
```python
from executorch.backends.qualcomm.utils.utils import (
    generate_qnn_executorch_compiler_spec,
    generate_htp_compiler_spec,
    QcomChipset,
    to_edge_transform_and_lower_to_qnn,
)

# HTP Compiler Configuration
backend_options = generate_htp_compiler_spec(
    use_fp16=not quantized,  # False for quantized models
)

# QNN Compiler Spec
compile_spec = generate_qnn_executorch_compiler_spec(
    soc_model=QcomChipset.SM8650,  # Your target SoC
    backend_options=backend_options,
)
```
#### Step 4: Lower and Export the Model
```python
# Lower to QNN backend
delegated_program = to_edge_transform_and_lower_to_qnn(
    quantized_model if quantized else model,
    example_inputs,
    compile_spec
)

# Export to ExecuTorch format
executorch_program = delegated_program.to_executorch()

# Save the compiled model
model_name = "custom_model_qnn.pte"
with open(model_name, "wb") as f:
    f.write(executorch_program.buffer)
print(f"Model successfully exported to {model_name}")
```

## Deep Dive

### Partitioner API

The **QnnPartitioner** identifies and groups supported subgraphs for execution on the QNN backend.  
It uses `QnnOperatorSupport` to check node-level compatibility with the Qualcomm backend via QNN SDK APIs.

The partitioner tags supported nodes with a `delegation_tag` and handles constants, buffers, and mutable states appropriately.
Please checkout [QNNPartitioner](https://github.com/pytorch/executorch/blob/main/backends/qualcomm/partition/qnn_partitioner.py#L125) for the latest changes. It mostly supports the following 4 inputs, and only compile spec is required
```python
class QnnPartitioner(Partitioner):
    """
    QnnPartitioner identifies subgraphs that can be lowered to QNN backend, by tagging nodes for delegation,
    and manages special cases such as mutable buffers and consumed constants.
    """

    def __init__(
        self,
        compiler_specs: List[CompileSpec],
        skip_node_id_set: set = None,
        skip_node_op_set: set = None,
        skip_mutable_buffer: bool = False,
    ):
        ...
```

### Quantization
Quantization in the QNN backend supports multiple data bit-widths and training modes (PTQ/QAT).
The QnnQuantizer defines quantization configurations and annotations compatible with Qualcomm hardware.

Supported schemes include:
- 8a8w (default)
- 16a16w
- 16a8w
- 16a4w
- 16a4w_block


Highlights:
- QuantDtype enumerates bit-width combinations for activations and weights.
- ModuleQConfig manages per-layer quantization behavior and observers.
- QnnQuantizer integrates with PT2E prepare/convert flow to annotate and quantize models.

Supports:

- Per-channel and per-block quantization

- Custom quant annotation via custom_quant_annotations

- Skipping specific nodes or ops

- Per-module customization via submodule_qconfig_list

For details, see: backends/qualcomm/quantizer/quantizer.py

### Operator Support
[The full operator support matrix](https://github.com/pytorch/executorch/tree/f32cdc3de6f7176d70a80228f1a60bcd45d93437/backends/qualcomm/builders#operator-support-status is tracked and frequently updated in the ExecuTorch repository.

It lists:
- Supported PyTorch ops (aten.*, custom ops)
- Planned ops
- Deprecated ops

This matrix directly corresponds to the implementations in: [executorch/backends/qualcomm/builders/node_visitors/*.py](https://github.com/pytorch/executorch/tree/main/backends/qualcomm/builders)

### Custom Ops Support

You can extend QNN backend support for your own operators.
Follow the [tutorial](https://github.com/pytorch/executorch/tree/f32cdc3de6f7176d70a80228f1a60bcd45d93437/examples/qualcomm/custom_op#custom-operator-support):

It covers:
- Writing new NodeVisitor for your op
- Registering via @register_node_visitor
- Creating and linking libQnnOp*.so for the delegate
- Testing and verifying custom kernels on HTP

## FAQ

If you encounter any issues while reproducing the tutorial, please file a github
[issue](https://github.com/pytorch/executorch/issues) on ExecuTorch repo and tag use `#qcom_aisw` tag

 ### Debugging tips
 - Before trying any complicated models, try out [a simple model example](https://github.com/pytorch/executorch/tree/f32cdc3de6f7176d70a80228f1a60bcd45d93437/examples/qualcomm#simple-examples-to-verify-the-backend-is-working) and see it if works one device.
