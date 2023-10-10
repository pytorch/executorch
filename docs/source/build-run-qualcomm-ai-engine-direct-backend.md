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

# Building and Running ExecuTorch on Qualcomm AI Engine Direct

In this tutorial we will walk you through the process of getting setup to
build ExecuTorch for Qualcomm AI Engine Direct and running a model on it.

Qualcomm AI Engine Direct is also referred to as QNN in the source and documentation.

<!----This will show a grid card on the page----->
::::{grid} 2
:::{grid-item-card}  What you will learn in this tutorial:
:class-card: card-learn
* In this tutorial you will learn how to lower and deploy a model for Qualcomm AI Engine Direct.
:::
:::{grid-item-card}  Tutorials we recommend you complete before this:
:class-card: card-prerequisites
* [Introduction to ExecuTorch](intro-how-it-works.md)
* [Setting up ExecuTorch](getting-started-setup.md)
* [Building ExecuTorch with CMake](runtime-build-and-cross-compilation.md)
:::
::::

## Prerequsites (Hardware and Software)

### Host OS

The Linux host operating system that QNN Backend is verified with is Ubuntu 20.04 LTS x64.

### Hardware:
You will need an Android smartphone with adb-connected running on one of below Qualcomm SoCs:
 - SM8450 (Snapdragon 8 Gen 1)
 - SM8475 (Snapdragon 8 Gen 1+)
 - SM8550 (Snapdragon 8 Gen 2)

This example is verified with SM8550.

### Software:

 - Follow ExecuTorch recommneded Python version.
 - A compiler to compile AOT parts. GCC 9.4 come with Ubuntu20.04 is verified.
 - Android NDK. This example is verified with NDK 25c.
 - [Qualcomm AI Engine Direct SDK](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk)
   - Follow the download button. After logging in, search Qualcomm AI Stack at the *Tool* panel.
   - You can find Qualcomm AI Engine Direct SDK under the AI Stack group.
   - Please download the Linux version, and follow instructions on the page to extract the file.
   - The SDK should be installed to somewhere `/opt/qcom/aistack/qnn` by default.
   - It's also OK to place it somewhere else. We don't have assumption about the absolute path of the SDK.
   - This example is verified with version 2.12.0.


## Setting up your developer environment

### Conventions

`$QNN_SDK_ROOT` refers to the root of Qualcomm AI Engine Direct SDK,
i.e., the directory containing `QNN_README.txt`.

`$ANDROID_NDK` refers to the root of Android NDK.

`$EXECUTORCH_ROOT` refers to the root of executorch git repository.

### Setup environment variables

We set `LD_LIBRARY_PATH` to make sure the dynamic linker can find QNN libraries.

Further, we set `PYTHONPATH` because it's easier to develop and import ExecuTorch
Pytho APIs.

```bash
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang/:$LD_LIBRARY_PATH
export PYTHONPATH=$EXECUTORCH_ROOT/..
```

## Build

An example script for below building instructions is [here](../../backends/qualcomm/scripts/build.sh).

### AOT (Ahead-of-time) components:

Python APIs on x64 are required to compile models to Qualcomm AI Engine Direct binary.

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

### Runtime:

A example `qnn_executor_runner` executable would be used to run the compiled `pte` model.

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

You can find `qnn_executor_runner` under `build_android/examples/qualcomm/`.


## Deploying and running on device

### AOT compile a model

You can refer to [this script](../../examples/qualcomm/scripts/export_example.py) for exact APIs.
We use mobilenet v2 as an example in this tutorial. Run below commands to get
a quantized mobilenetv2 delegated to QNN:

```
python -m examples.qualcomm.scripts.export_example --model_name mv2
```

A file called `mv2.pte` would be generated.


### Run model Inference on an Android smartphone with Qualcomm SoCs

***Step 1***. We need to push required QNN libraries to the device.

```bash
# make sure you have write-permission on below path.
DEVICE_DIR=/data/local/tmp/executorch_qualcomm_tutorial/
adb shell "mkdir -p ${DEVICE_DIR}"
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV69Stub.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV73Stub.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnSystem.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/hexagon-v69/unsigned/libQnnHtpV69Skel.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so ${DEVICE_DIR}
```

***Step 2***.  We also need to indicate dynamic linkers on Android and Hexagon
where to find these libraries by setting `ADSP_LIBRARY_PATH` and `LD_LIBRARY_PATH`.
So, we can run `qnn_executor_runner` like

```bash
adb push mv2.pte ${DEVICE_DIR}
adb push ${EXECUTORCH_ROOT}/build_android/examples/qualcomm/qnn_executor_runner ${DEVICE_DIR}
adb shell "cd ${DEVICE_DIR} \
           && export LD_LIBRARY_PATH=${DEVICE_DIR} \
           && export ADSP_LIBRARY_PATH=${DEVICE_DIR} \
           && ./qnn_executor_runner --model_path ./mv2_qnn.pte"
```

You should see the following result.
Note that no output file will be generated in this example.
```
I 00:00:00.133366 executorch:qnn_executor_runner.cpp:156] Method loaded.
I 00:00:00.133590 executorch:util.h:104] input already initialized, refilling.
I 00:00:00.135162 executorch:qnn_executor_runner.cpp:161] Inputs prepared.
I 00:00:00.136768 executorch:qnn_executor_runner.cpp:278] Model executed successfully.
[INFO][Qnn ExecuTorch] Destroy Qnn backend parameters
[INFO][Qnn ExecuTorch] Destroy Qnn context
[INFO][Qnn ExecuTorch] Destroy Qnn device
[INFO][Qnn ExecuTorch] Destroy Qnn backend
```

