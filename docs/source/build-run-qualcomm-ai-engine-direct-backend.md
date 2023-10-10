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

This example is verified with SM8550 and SM8450.

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
├── qaisw-v2.13.1.230725130242_60412
├── QNN_NOTICE.txt
├── QNN_README.txt
├── QNN_ReleaseNotes.txt
├── QNN_SECUREPD_README.txt
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

You can refer to [this script](../../examples/qualcomm/scripts/deeplab_v3.py) for the exact flow.
We use deeplab-v3-resnet101 as an example in this tutorial. Run below commands to see how an
end-to-end example works:

```
cd $EXECUTORCH_ROOT
python -m examples.qualcomm.scripts.deeplab_v3 -b build_android -m SM8550 -s <adb_connected_device_serial>
```

All artifacts would be put to a directory `./deeplab_v3`, including a compiled pte file, `dlv3_qnn.pte`.


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
Model outputs are put to `./deeplab_v3/outputs`.
```
I 00:00:01.863385 executorch:qnn_executor_runner.cpp:298] 100 inference took 1096.624000 ms, avg 10.966240 ms
[INFO][Qnn ExecuTorch] Destroy Qnn backend parameters
[INFO][Qnn ExecuTorch] Destroy Qnn context
[INFO][Qnn ExecuTorch] Destroy Qnn device
[INFO][Qnn ExecuTorch] Destroy Qnn backend
[WARNING] <W> Cannot get op package maps as singleton provider is already destroyed

/data/local/tmp/executorch/dlv3_qnn/outputs/: 200 files pulled. 46.7 MB/s (842956800 bytes in 17.198s)
PA   : 0.9279080163742056%
MPA  : 0.8436995479010411%
MIoU : 0.7295181225907508%
CIoU :
{
  "Backround": 0.9118397702257772,
  "Aeroplane": 0.9195003650537316,
  "Bicycle": 0.4074836630439685,
  "Bird": 0.8318192515690322,
  "Boat": 0.5604090467931566,
  "Bottle": 0.810288138560711,
  "Bus": 0.9462156473522969,
  "Car": 0.9036561912572266,
  "Cat": 0.9218752933579742,
  "Chair": 0.166601521782278,
  "Cow": 0.9255118338630605,
  "DiningTable": 0.7566001181986504,
  "Dog": 0.7315379195337196,
  "Horse": 0.6748639306871056,
  "MotorBike": 0.8083967507343583,
  "Person": 0.8554745611396247,
  "PottedPlant": 0.40625618681325354,
  "Sheep": 0.8037552530978187,
  "Sofa": 0.47743485657288254,
  "Train": 0.6377080913628297,
  "TvMonitor": 0.8626521834063088
}
```


### Running a model via ExecuTorch's android demo-app

An Android demo-app using Qualcomm AI Engine Direct Backend can be found in
`examples`. Please refer to [README.md](../../examples/demo-apps/android/ExecuTorchDemo/README.md).


### What is coming?

 - [An example using quantized mobilebert](https://github.com/pytorch/executorch/pull/658) to solve multi-class text classification.
 - More Qualcomm AI Engine Direct accelerators, e.g., GPU.

