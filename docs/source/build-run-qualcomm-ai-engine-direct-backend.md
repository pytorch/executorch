# Building and Running ExecuTorch with Qualcomm AI Engine Direct Backend

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
* [Setting up ExecuTorch](getting-started-setup.md)
* [Building ExecuTorch with CMake](runtime-build-and-cross-compilation.md)
:::
::::


## What's Qualcomm AI Engine Direct?

[Qualcomm AI Engine Direct](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk)
is designed to provide unified, low-level APIs for AI development.

Developers can interact with various accelerators on Qualcomm SoCs with these set of APIs, including
Kryo CPU, Adreno GPU, and Hexagon processors. More details can be found [here](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/overview.html).

Currently, this ExecuTorch Backend can delegate AI computations to Hexagon processors through Qualcomm AI Engine Direct APIs.


## Prerequsites (Hardware and Software)

### Host OS

The Linux host operating system that QNN Backend is verified with is Ubuntu 22.04 LTS x64
at the moment of updating this tutorial.
Usually, we verified the backend on the same OS version which QNN is verified with.
The version is documented in QNN SDK.

### Hardware:
You will need an Android smartphone with adb-connected running on one of below Qualcomm SoCs:
 - SM8450 (Snapdragon 8 Gen 1)
 - SM8475 (Snapdragon 8 Gen 1+)
 - SM8550 (Snapdragon 8 Gen 2)
 - SM8650 (Snapdragon 8 Gen 3)

This example is verified with SM8550 and SM8450.

### Software:

 - Follow ExecuTorch recommended Python version.
 - A compiler to compile AOT parts, e.g., the GCC compiler comes with Ubuntu LTS.
 - [Android NDK](https://developer.android.com/ndk). This example is verified with NDK 26c.
 - [Qualcomm AI Engine Direct SDK](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk)
   - Click the "Get Software" button to download a version of QNN SDK.
   - However, at the moment of updating this tutorial, the above website doesn't provide QNN SDK newer than 2.22.6.
   - The below is public links to download various QNN versions. Hope they can be publicly discoverable soon.
   - [QNN 2.26.0](https://softwarecenter.qualcomm.com/api/download/software/qualcomm_neural_processing_sdk/v2.26.0.240828.zip)

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
The above script is actively used. It is updated more frquently than this tutorial.
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
  -DPYTHON_EXECUTABLE=python3 \
  -DEXECUTORCH_SEPARATE_FLATCC_HOST_PROJECT=OFF

# nproc is used to detect the number of available CPU.
# If it is not applicable, please feel free to use the number you want.
cmake --build $PWD --target "PyQnnManagerAdaptor" "PyQnnWrapperAdaptor" -j$(nproc)

# install Python APIs to correct import path
# The filename might vary depending on your Python and host version.
cp -f backends/qualcomm/PyQnnManagerAdaptor.cpython-310-x86_64-linux-gnu.so $EXECUTORCH_ROOT/backends/qualcomm/python
cp -f backends/qualcomm/PyQnnWrapperAdaptor.cpython-310-x86_64-linux-gnu.so $EXECUTORCH_ROOT/backends/qualcomm/python

# Workaround for fbs files in exir/_serialize
cp $EXECUTORCH_ROOT/schema/program.fbs $EXECUTORCH_ROOT/exir/_serialize/program.fbs
cp $EXECUTORCH_ROOT/schema/scalar_type.fbs $EXECUTORCH_ROOT/exir/_serialize/scalar_type.fbs
```

### Runtime:

A example `qnn_executor_runner` executable would be used to run the compiled `pte` model.

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
    -DANDROID_NATIVE_API_LEVEL=23

# nproc is used to detect the number of available CPU.
# If it is not applicable, please feel free to use the number you want.
cmake --build $PWD --target install -j$(nproc)

cmake ../examples/qualcomm \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI='arm64-v8a' \
    -DANDROID_NATIVE_API_LEVEL=23 \
    -DCMAKE_PREFIX_PATH="$PWD/lib/cmake/ExecuTorch;$PWD/third-party/gflags;" \
    -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
    -DPYTHON_EXECUTABLE=python3 \
    -Bexamples/qualcomm

cmake --build examples/qualcomm -j$(nproc)

# qnn_executor_runner can be found under examples/qualcomm
# The full path is $EXECUTORCH_ROOT/build-android/examples/qualcomm/qnn_executor_runner
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

# qnn_executor_runner can be found under examples/qualcomm
# The full path is $EXECUTORCH_ROOT/build-x86/examples/qualcomm/qnn_executor_runner
ls examples/qualcomm/
```

To run the HTP emulator, the dynamic linker need to access QNN libraries and `libqnn_executorch_backend.so`.
We set the below two paths to `LD_LIBRARY_PATH` environment variable:
  1. `$QNN_SDK_ROOT/lib/x86_64-linux-clang/`
  2. `$EXECUTORCH_ROOT/build-x86/lib/`

The first path is for QNN libraries including HTP emulator. It has been configured in the AOT compilation section.

The second path is for `libqnn_executorch_backend.so`.

So, we can run `./deeplab_v3/dlv3_qnn.pte` by:
```bash
cd $EXECUTORCH_ROOT/build-x86
export LD_LIBRARY_PATH=$EXECUTORCH_ROOT/build-x86/lib/:$LD_LIBRARY_PATH
examples/qualcomm/qnn_executor_runner --model_path ../deeplab_v3/dlv3_qnn.pte
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
adb push ${QNN_SDK_ROOT}/lib/hexagon-v69/unsigned/libQnnHtpV69Skel.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so ${DEVICE_DIR}
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

The command-line arguents are written in [utils.py](https://github.com/pytorch/executorch/blob/main/examples/qualcomm/scripts/utils.py#L127).
The model, inputs, and output location are passed to `qnn_executorch_runner` by `--model_path`, `--input_list_path`, and `--output_folder_path`.


### Running a model via ExecuTorch's android demo-app

An Android demo-app using Qualcomm AI Engine Direct Backend can be found in
`examples`. Please refer to android demo app [tutorial](https://pytorch.org/executorch/stable/demo-apps-android.html).

## Supported model list

Please refer to `$EXECUTORCH_ROOT/examples/qualcomm/scripts/` and `EXECUTORCH_ROOT/examples/qualcomm/oss_scripts/` to the list of supported models.

## What is coming?

 - Improve the performance for llama3-8B-Instruct and support batch prefill.
 - We will support pre-compiled binaries from [Qualcomm AI Hub](https://aihub.qualcomm.com/).

## FAQ

If you encounter any issues while reproducing the tutorial, please file a github
issue on ExecuTorch repo and tag use `#qcom_aisw` tag
