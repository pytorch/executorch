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

The Linux host operating system that QNN Backend is verified with is Ubuntu 20.04 LTS x64.

However, because Qualcomm Package Manager(QPM) used to download necessary SDK (see below)
only support Ubuntu, we recommend users to exercise this tutorial exacly
on Ubuntu 20.04.

### Hardware:
You will need an Android smartphone with adb-connected running on one of below Qualcomm SoCs:
 - SM8450 (Snapdragon 8 Gen 1)
 - SM8475 (Snapdragon 8 Gen 1+)
 - SM8550 (Snapdragon 8 Gen 2)
 - SM8650 (Snapdragon 8 Gen 3)

This example is verified with SM8550 and SM8450.

### Software:

 - Follow ExecuTorch recommended Python version.
 - A compiler to compile AOT parts. GCC 9.4 come with Ubuntu20.04 is verified.
 - [Android NDK](https://developer.android.com/ndk). This example is verified with NDK 25c.
 - [Qualcomm AI Engine Direct SDK](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk)
   - Follow the download button. After logging in, search Qualcomm AI Stack at the *Tool* panel.
   - You can find Qualcomm AI Engine Direct SDK under the AI Stack group.
   - Please download the Linux version, and follow instructions on the page to extract the file.
   - The SDK should be installed to somewhere `/opt/qcom/aistack/qnn` by default.
   - It's also OK to place it somewhere else. We don't have assumption about the absolute path of the SDK.
   - This example is verified with version 2.12.0.

The directory with installed Qualcomm AI Engine Direct SDK looks like:
```
$ tree -L 1 /opt/qcom/aistack/qnn/<version>/
/opt/qcom/aistack/qnn/<version>/
├── benchmarks
├── bin
├── docs
├── examples
├── include
├── lib
├── LICENSE.pdf
├── QNN_NOTICE.txt
├── QNN_README.txt
├── QNN_ReleaseNotes.txt
├── share
└── Uninstall
```


## Setting up your developer environment

### Conventions

`$QNN_SDK_ROOT` refers to the root of Qualcomm AI Engine Direct SDK,
i.e., the directory containing `QNN_README.txt`.

`$ANDROID_NDK` refers to the root of Android NDK.

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

An example script for below building instructions is [here](https://github.com/pytorch/executorch/blob/main/backends/qualcomm/scripts/build.sh).

### AOT (Ahead-of-time) components:

Python APIs on x64 are required to compile models to Qualcomm AI Engine Direct binary.

```bash
cd $EXECUTORCH_ROOT
# Workaround for fbs files in exir/_serialize
cp schema/program.fbs exir/_serialize/program.fbs
cp schema/scalar_type.fbs exir/_serialize/scalar_type.fbs

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
# build executorch & qnn_executorch_backend
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$PWD \
    -DEXECUTORCH_BUILD_SDK=ON \
    -DEXECUTORCH_BUILD_QNN=ON \
    -DQNN_SDK_ROOT=$QNN_SDK_ROOT \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI='arm64-v8a' \
    -DANDROID_NATIVE_API_LEVEL=23 \
    -B$PWD

cmake --build $PWD -j16 --target install

cmake ../examples/qualcomm \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI='arm64-v8a' \
    -DANDROID_NATIVE_API_LEVEL=23 \
    -DCMAKE_PREFIX_PATH="$PWD/lib/cmake/ExecuTorch;$PWD/third-party/gflags;" \
    -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
    -Bexamples/qualcomm

cmake --build examples/qualcomm -j16
```

**Note:** If you want to build for release, add `-DCMAKE_BUILD_TYPE=Release` to the `cmake` command options.

You can find `qnn_executor_runner` under `build_android/examples/qualcomm/`.


## Deploying and running on device

### AOT compile a model

You can refer to [this script](https://github.com/pytorch/executorch/blob/main/examples/qualcomm/scripts/deeplab_v3.py) for the exact flow.
We use deeplab-v3-resnet101 as an example in this tutorial. Run below commands to compile:

```
cd $EXECUTORCH_ROOT
python -m examples.qualcomm.scripts.deeplab_v3 -b build_android -m SM8550 --compile_only --download
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


### Run model inference on an Android smartphone with Qualcomm SoCs

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
adb push ./deeplab_v3/dlv3_qnn.pte ${DEVICE_DIR}
adb push ${EXECUTORCH_ROOT}/build_android/examples/qualcomm/qnn_executor_runner ${DEVICE_DIR}
adb push ${EXECUTORCH_ROOT}/build_android/lib/libqnn_executorch_backend.so ${DEVICE_DIR}
adb shell "cd ${DEVICE_DIR} \
           && export LD_LIBRARY_PATH=${DEVICE_DIR} \
           && export ADSP_LIBRARY_PATH=${DEVICE_DIR} \
           && ./qnn_executor_runner --model_path ./dlv3_qnn.pte"
```

You should see something like below:

```
I 00:00:01.835706 executorch:qnn_executor_runner.cpp:298] 100 inference took 1096.626000 ms, avg 10.966260 ms
[INFO][Qnn ExecuTorch] Destroy Qnn backend parameters
[INFO][Qnn ExecuTorch] Destroy Qnn context
[INFO][Qnn ExecuTorch] Destroy Qnn device
[INFO][Qnn ExecuTorch] Destroy Qnn backend
```


### Running a model via ExecuTorch's android demo-app

An Android demo-app using Qualcomm AI Engine Direct Backend can be found in
`examples`. Please refer to android demo app [tutorial](https://pytorch.org/executorch/stable/demo-apps-android.html).


## What is coming?

 - [An example using quantized mobilebert](https://github.com/pytorch/executorch/pull/1043) to solve multi-class text classification.
 - More Qualcomm AI Engine Direct accelerators, e.g., GPU.

## FAQ

If you encounter any issues while reproducing the tutorial, please file a github
issue on ExecuTorch repo and tag use `#qcom_aisw` tag
