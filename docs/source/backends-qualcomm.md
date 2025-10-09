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

The Linux host operating system that QNN Backend is verified with is Ubuntu 22.04 LTS x64
at the moment of updating this tutorial.
In addition, it is also confirmed to work on Windows Subsystem for Linux (WSL) with Ubuntu 22.04.
Usually, we verified the backend on the same OS version which QNN is verified with.
The version is documented in QNN SDK.

#### Windows (WSL) Setup
To install Ubuntu 22.04 on WSL, run the following command in PowerShell or Windows Terminal:
``` bash
wsl --install -d ubuntu 22.04
```
This command will install WSL and set up Ubuntu 22.04 as the default Linux distribution.

For more details and troubleshooting, refer to the official Microsoft WSL installation guide:
👉 [Install WSL | Microsoft Learn](https://learn.microsoft.com/en-us/windows/wsl/install)

### Hardware:
You will need an Android smartphone with adb-connected running on one of below Qualcomm SoCs:
 - SA8295
 - SM8450 (Snapdragon 8 Gen 1)
 - SM8475 (Snapdragon 8 Gen 1+)
 - SM8550 (Snapdragon 8 Gen 2)
 - SM8650 (Snapdragon 8 Gen 3)
 - SM8750 (Snapdragon 8 Elite)
 - SSG2115P
 - SSG2125P
 - SXR1230P
 - SXR2230P
 - SXR2330P

This example is verified with SM8550 and SM8450.

### Software:

 - Follow ExecuTorch recommended Python version.
 - A compiler to compile AOT parts, e.g., the GCC compiler comes with Ubuntu LTS.
 - [Android NDK](https://developer.android.com/ndk). This example is verified with NDK 26c.
 - [Qualcomm AI Engine Direct SDK](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk)
   - Click the "Get Software" button to download a version of QNN SDK.
   - However, at the moment of updating this tutorial, the above website doesn't provide QNN SDK newer than 2.22.6.
   - The below is public links to download various QNN versions. Hope they can be publicly discoverable soon.
   - [QNN 2.37.0](https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.37.0.250724/v2.37.0.250724.zip)

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
./backends/qualcomm/scripts/build.sh
# or
./backends/qualcomm/scripts/build.sh --release
```

### AOT (Ahead-of-time) components:

Python APIs on x64 are required to compile models to Qualcomm AI Engine Direct binary.

```bash
cd $EXECUTORCH_ROOT
mkdir build-x86
cd build-x86
# Note that the below command might change.
# Please refer to the above build.sh for latest workable commands.
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$PWD \
  -DEXECUTORCH_BUILD_QNN=ON \
  -DQNN_SDK_ROOT=${QNN_SDK_ROOT} \
  -DEXECUTORCH_BUILD_DEVTOOLS=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
  -DPYTHON_EXECUTABLE=python3

# nproc is used to detect the number of available CPU.
# If it is not applicable, please feel free to use the number you want.
cmake --build $PWD --target "PyQnnManagerAdaptor" "PyQnnWrapperAdaptor" -j$(nproc)

# install Python APIs to correct import path
# The filename might vary depending on your Python and host version.
cp -f backends/qualcomm/PyQnnManagerAdaptor.cpython-310-x86_64-linux-gnu.so $EXECUTORCH_ROOT/backends/qualcomm/python
cp -f backends/qualcomm/PyQnnWrapperAdaptor.cpython-310-x86_64-linux-gnu.so $EXECUTORCH_ROOT/backends/qualcomm/python

# Workaround for .fbs files in exir/_serialize
cp $EXECUTORCH_ROOT/schema/program.fbs $EXECUTORCH_ROOT/exir/_serialize/program.fbs
cp $EXECUTORCH_ROOT/schema/scalar_type.fbs $EXECUTORCH_ROOT/exir/_serialize/scalar_type.fbs
```

### Runtime:

An example `qnn_executor_runner` executable would be used to run the compiled `pte` model.

Commands to build `qnn_executor_runner` for Android:

```bash
cd $EXECUTORCH_ROOT
mkdir build-android
cd build-android
# build executorch & qnn_executorch_backend
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$PWD \
    -DEXECUTORCH_BUILD_QNN=ON \
    -DQNN_SDK_ROOT=$QNN_SDK_ROOT \
    -DEXECUTORCH_BUILD_DEVTOOLS=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
    -DPYTHON_EXECUTABLE=python3 \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI='arm64-v8a' \
    -DANDROID_PLATFORM=android-30

# nproc is used to detect the number of available CPU.
# If it is not applicable, please feel free to use the number you want.
cmake --build $PWD --target install -j$(nproc)

cmake ../examples/qualcomm \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI='arm64-v8a' \
    -DANDROID_PLATFORM=android-30 \
    -DCMAKE_PREFIX_PATH="$PWD/lib/cmake/ExecuTorch;$PWD/third-party/gflags;" \
    -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
    -DPYTHON_EXECUTABLE=python3 \
    -Bexamples/qualcomm

cmake --build examples/qualcomm -j$(nproc)

# qnn_executor_runner can be found under examples/qualcomm
# The full path is $EXECUTORCH_ROOT/build-android/examples/qualcomm/executor_runner/qnn_executor_runner
ls examples/qualcomm
```

**Note:** If you want to build for release, add `-DCMAKE_BUILD_TYPE=Release` to the `cmake` command options.


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
adb push ${QNN_SDK_ROOT}/lib/hexagon-v75/unsigned/libQnnHtpV79Skel.so ${DEVICE_DIR}
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
python -m examples.qualcomm.scripts.deeplab_v3 -b build-android -m SM8550 --download -s <device_serial>
```
The `<device_serial>` can be found by `adb devices` command.

After the above command, pre-processed inputs and outputs are put in `$EXECUTORCH_ROOT/deeplab_v3` and `$EXECUTORCH_ROOT/deeplab_v3/outputs` folder.

The command-line arguments are written in [utils.py](https://github.com/pytorch/executorch/blob/main/examples/qualcomm/utils.py#L139).
The model, inputs, and output location are passed to `qnn_executorch_runner` by `--model_path`, `--input_list_path`, and `--output_folder_path`.


## Supported model list

Please refer to `$EXECUTORCH_ROOT/examples/qualcomm/scripts/` and `EXECUTORCH_ROOT/examples/qualcomm/oss_scripts/` to the list of supported models.

## How to Support a Custom Model in HTP Backend

### Step-by-Step Implementation Guide

Please reference [the simple example](https://github.com/pytorch/executorch/blob/main/examples/qualcomm/scripts/export_example.py) and [more compilated examples](https://github.com/pytorch/executorch/tree/main/examples/qualcomm/scripts) for reference
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

## What is coming?

 - Improve the performance for llama3-8B-Instruct and support batch prefill.
 - We will support pre-compiled binaries from [Qualcomm AI Hub](https://aihub.qualcomm.com/).

## FAQ

If you encounter any issues while reproducing the tutorial, please file a github
[issue](https://github.com/pytorch/executorch/issues) on ExecuTorch repo and tag use `#qcom_aisw` tag
