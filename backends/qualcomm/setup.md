# Setting up QNN Backend

This is a tutorial for building and running Qualcomm AI Engine Direct backend,
including compiling a model on a x64 host and running the inference
on a Android device.


## Prerequisite

Please finish tutorial [Setting up executorch](../../docs/source/getting-started-setup.md).


## Conventions

`$QNN_SDK_ROOT` refers to the root of Qualcomm AI Engine Direct SDK,
i.e., the directory containing `QNN_README.txt`.

`$ANDROID_NDK_ROOT` refers to the root of Android NDK.

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

Note: Since we set `PYTHONPATH`, we may have issue with finding `program.fbs`
and `scalar_type.fbs` when we export a model, because they are installed into
`pip-out` directory with the same package name pattern. A workaround is that
we copy `$EXECUTORCH_ROOT/pip-out/lib.linux-x86_64-cpython-310/executorch/exir/_serialize/program.fbs`
and `$EXECUTORCH_ROOT/pip-out/lib.linux-x86_64-cpython-310/executorch/exir/_serialize/scalar_type.fbs`
to `$EXECUTORCH_ROOT/exir/_serialize/`.


## End to End Inference

### Step 1: Build Python APIs for AOT compilation on x64

Python APIs on x64 are required to compile models to Qualcomm AI Engine Direct binary.
Make sure `buck2` is under a directory in `PATH`.

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
# build executorch & qnn_executorch_backend
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$PWD \
    -DEXECUTORCH_BUILD_QNN=ON \
    -DEXECUTORCH_BUILD_SDK=ON \
    -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
    -DQNN_SDK_ROOT=$QNN_SDK_ROOT \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI='arm64-v8a' \
    -DANDROID_NATIVE_API_LEVEL=23 \
    -B$PWD

cmake --build $PWD -j16 --target install

cmake ../examples/qualcomm \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI='arm64-v8a' \
    -DANDROID_NATIVE_API_LEVEL=23 \
    -DCMAKE_PREFIX_PATH="$PWD/lib/cmake/ExecuTorch;$PWD/third-party/gflags;" \
    -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
    -Bexamples/qualcomm

cmake --build examples/qualcomm -j16
```
**Note:** If you want to build for release, add `-DCMAKE_BUILD_TYPE=Release` to the `cmake` command options.

You can find `qnn_executor_runner` under `build_android/examples/qualcomm/`.


### Step 3: Compile a model

```
python -m examples.qualcomm.scripts.export_example --model_name mv2
```

Then the generated `mv2.pte` can be run on the device by
`build_android/backends/qualcomm/qnn_executor_runner` with Qualcomm AI Engine
Direct backend.

[**Note**] To get proper accuracy, please apply calibrations with representative
dataset, which could be learnt more from examples under `examples/qualcomm/`.


### Step 4: Model Inference

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
