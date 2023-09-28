# Setting up QNN Backend

This is a tutorial for building and running Qualcomm AI Engine Direct backend,
including compiling a model on a x64 host and running the inference
on a Android device.


## Prerequisite

Please finish tutorial [Setting up executorch](../../docs/website/docs/tutorials/00_setting_up_executorch.md).


## Conventions

`$QNN_SDK_ROOT` refers to the root of Qualcomm AI Engine Direct SDK,
i.e., the directory containing `QNN_README.txt`.

`$ANDROID_NDK` refers to the root of Android NDK.

`$EXECUTORCH_ROOT` refers to the root of executorch git repository.


## Environment Setup

### Download Qualcomm AI Engine Direct SDK

Navigate to [Qualcomm AI Engine Direct SDK](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk) and follow the download button.

You might need to apply for a Qualcomm account to download the SDK.

After logging in, search Qualcomm AI Stack at the *Tool* panel.
You can find Qualcomm AI Engine Direct SDK under the AI Stack group.

Please download the Linux version, and follow instructions on the page to
extract the file.

The SDK should be installed to somewhere `/opt/qcom/aistack/qnn` by default.

### Download Android NDK

Please navigate to [Android NDK](https://developer.android.com/ndk) and download
a version of NDK. We recommend LTS version, currently r25c.

### Setup environment variables

We need to make sure Qualcomm AI Engine Direct libraries can be found by
the dynamic linker on x64. Hence we set `LD_LIBRARY_PATH`. In production,
we recommend users to put libraries in default search path or use `rpath`
to indicate the location of libraries.

Further, we set up `$PYTHONPATH` because it's easier to develop and import executorch Python APIs. Users might also build and install executorch package as usual python package.

```bash
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang/:$LD_LIBRARY_PATH
export PYTHONPATH=$EXECUTORCH_ROOT/..
```


## Build

### Step 1: Build Python APIs for AOT compilation on x64

Python APIs on x64 are required to compile models to Qualcomm AI
Make sure `buck2` is under a directory in `PATH`.
Engine Direct binary.

```bash
cd $EXECUTORCH_ROOT
mkdir build_x86_64
cd build_x86_64
cmake .. -DEXECUTORCH_BUILD_QNN=ON -DQNN_SDK_ROOT=${QNN_SDK_ROOT}
cmake --build . -t "PyQnnManagerAdaptor" "PyQnnWrapperAdaptor" -j8

# install Python APIs to correct import path
# The filename might vary depending on your Python and host version.
cp -f backends/qualcomm/PyQnnManagerAdaptor.cpython-310-x86_64-linux-gnu.so $EXECUTORCH_ROOT/backends/qualcomm/python
cp -f backends/qualcomm/PyQnnWrapperAdaptor.cpython-310-x86_64-linux-gnu.so $EXECUTORCH_ROOT/backends/qualcomm/python
```


### Step 2: Build `qnn_executor_runner` for Android

`qnn_executor_runner` is an executable running the compiled model.

You might want to ensure the correct `flatc`. `flatc` can be built along with the above step. For example, we can find `flatc` in `build_x86_64/third-party/flatbuffers/`.

We can prepend `$EXECUTORCH_ROOT/build_x86_64/third-party/flatbuffers` to `PATH`. Then below cross-compiling can find the correct flatbuffer compiler.

Commands to build `qnn_executor_runner` for Android:

```bash
cd $EXECUTORCH_ROOT
mkdir build_android
cd build_android
cmake .. -DQNN_SDK_ROOT=$QNN_SDK_ROOT \
    -DEXECUTORCH_BUILD_QNN=ON \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI='arm64-v8a' \
    -DANDROID_NATIVE_API_LEVEL=23 \
    -DBUCK2=buck2
cmake --build . -j8
```

You can find `qnn_executor_runner` under `build_android/backends/qualcomm`.


## Compile a model

### for ExecuTorch

We can export and compile an executorch model by:

```python
from executorch import exir
from executorch.examples.models.mobilenet_v2 import MV2Model

mv2 = MV2Model()
example_input = mv2.get_example_inputs()
model = mv2.get_eager_model().eval()

captured_program = exir.capture(model, example_input)

edge_program = captured_program.to_edge(
    exir.EdgeCompileConfig(_check_ir_validity=False))

executorch_program = edge_program.to_executorch()

with open("mv2.pte", "wb") as fp:
    fp.write(executorch_program.buffer)
```

Then the `mv2.pte` can be run on the device by `build_android/executor_runner`.

### for executorch Qualcomm AI Engine backend

So, how if we want to compile the model for Qualcomm AI Engine?

It's as simple as changing some configurations and adding an extra call `to_backend()`.

To gain the best performance, we also employ quantization:

```python
import torch

from torch.ao.quantization.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
)
from executorch.examples.models.mobilenet_v2 import MV2Model
from executorch.exir.backend.backend_api import (
    to_backend,
    validation_disabled,
)
from executorch.backends.qualcomm.partition.qnn_partitioner import (
    QnnPartitioner,
)
from executorch.backends.qualcomm.qnn_quantizer import (
    QnnQuantizer,
    get_default_qnn_ptq_config,
)
from executorch.backends.qualcomm.utils.utils import (
    capture_program,
    generate_qnn_executorch_compiler_spec,
)

mv2 = MV2Model()
example_input = mv2.get_example_inputs()
model = mv2.get_eager_model().eval()

# Get quantizer
quantizer = QnnQuantizer()
quant_config = get_default_qnn_ptq_config(enable_per_channel_conv_quant=False)
quantizer.set_global_op_quant_config(quant_config)

# Typical pytorch 2.0 quantization flow
m = torch._export.capture_pre_autograd_graph(model, example_input)
m = prepare_pt2e(m, quantizer)
# Calibration
m(*example_input)
# Get the quantized model
m = convert_pt2e(m)

# Capture program for edge IR
edge_program = capture_program(m, example_input)

# Delegate to QNN backend
QnnPartitioner.set_compiler_spec(
    generate_qnn_executorch_compiler_spec(
        is_fp16=False, soc_model="SM8550", debug=False, saver=False,
    )
)
with validation_disabled():
    delegated_program = edge_program
    delegated_program.exported_program = to_backend(
        edge_program.exported_program, QnnPartitioner
    )

executorch_program = delegated_program.to_executorch()

with open("mv2_qnn.pte", "wb") as fp:
    fp.write(executorch_program.buffer)
```

Then we have an executorch program which delegates the model to
Qualcomm AI Engine Direct backend.


## Run

The backend rely on Qualcomm AI Engine Direct SDK libraries.

You might want to follow docs in Qualcomm AI Engine Direct SDK to setup the device environment.
Or see below for a quick setup for testing:

```bash
# make sure you have write-permission on below path.
DEVICE_DIR=/data/local/tmp/executorch_test/
adb shell "mkdir -p ${DEVICE_DIR}"
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV69Stub.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV73Stub.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnSystem.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/hexagon-v69/unsigned/libQnnHtpV69Skel.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so ${DEVICE_DIR}
```

We also need to indicate dynamic linkers on Android and Hexagon where to find these libraries
by setting `ADSP_LIBRARY_PATH` and `LD_LIBRARY_PATH`.

So, we can run `qnn_executor_runner` like
```bash
adb push mv2_qnn.pte ${DEVICE_DIR}
adb push ${EXECUTORCH_ROOT}/build_android/backends/qualcomm/qnn_executor_runner ${DEVICE_DIR}
adb shell "cd ${DEVICE_DIR} \
           && export LD_LIBRARY_PATH=${DEVICE_DIR} \
           && export ADSP_LIBRARY_PATH=${DEVICE_DIR} \
           && ./qnn_executor_runner --model_path ./mv2_qnn.pte"
```

You should see the following result.
Note that no output file will be generated in this example.
```bash
I 00:00:00.133366 executorch:qnn_executor_runner.cpp:156] Method loaded.
I 00:00:00.133590 executorch:util.h:104] input already initialized, refilling.
I 00:00:00.135162 executorch:qnn_executor_runner.cpp:161] Inputs prepared.
I 00:00:00.136768 executorch:qnn_executor_runner.cpp:278] Model executed successfully.
[INFO][Qnn Execu Torch] Destroy Qnn backend parameters
[INFO][Qnn Execu Torch] Destroy Qnn context
[INFO][Qnn Execu Torch] Destroy Qnn device
[INFO][Qnn Execu Torch] Destroy Qnn backend
```
