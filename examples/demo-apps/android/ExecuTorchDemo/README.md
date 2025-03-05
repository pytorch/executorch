# Building an ExecuTorch Android Demo App

This is forked from [PyTorch Android Demo App](https://github.com/pytorch/android-demo-app).

This guide explains how to setup ExecuTorch for Android using a demo app. The app employs a [DeepLab v3](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/) model for image segmentation tasks. Models are exported to ExecuTorch using [XNNPACK FP32 backend](tutorial-xnnpack-delegate-lowering.md).

::::{grid} 2
:::{grid-item-card}  What you will learn
:class-card: card-prerequisites
* How to set up a build target for Android arm64-v8a
* How to build the required ExecuTorch runtime with JNI wrapper for Android
* How to build the app with required JNI library and model file
:::

:::{grid-item-card} Prerequisites
:class-card: card-prerequisites
* Refer to [Setting up ExecuTorch](https://pytorch.org/executorch/stable/getting-started-setup) to set up the repo and dev environment.
* Download and install [Android Studio and SDK](https://developer.android.com/studio).
* Supported Host OS: CentOS, macOS Ventura (M1/x86_64). See below for Qualcomm HTP specific requirements.
* *Qualcomm HTP Only[^1]:* To build and run on Qualcomm's AI Engine Direct, please follow [Building and Running ExecuTorch with Qualcomm AI Engine Direct Backend](backends-qualcomm.md) for hardware and software pre-requisites. The version we use for this tutorial is 2.19. The chip we use for this tutorial is SM8450.
:::
::::

[^1]: This section applies only if Qualcomm HTP Backend is needed in the app. Same applies to sections with title`Qualcomm Hexagon NPU`.

```{note}
This demo app and tutorial has only been validated with arm64-v8a [ABI](https://developer.android.com/ndk/guides/abis).
```


## Build

### Ahead-Of-Time

We generate the model file for the ExecuTorch runtime in Android Demo App.

#### XNNPACK Delegation

For delegating DeepLab v3 to XNNPACK backend, please do the following to export the model:

```bash
cd executorch # go to executorch root
python3 -m examples.xnnpack.aot_compiler --model_name="dl3" --delegate
```

Then push the pte file to Android device:

```bash
adb push dl3_xnnpack_fp32.pte /data/local/tmp/dl3_xnnpack_fp32.pte
```

For more detailed tutorial of lowering to XNNPACK, please see [XNNPACK backend](backends-xnnpack.md).

#### Qualcomm Hexagon NPU

For delegating to Qualcomm Hexagon NPU, please follow the tutorial [here](backends-qualcomm.md).

```bash
python -m examples.qualcomm.scripts.deeplab_v3 -b build-android -m SM8450 -s <adb_connected_device_serial>
```

Then push the pte file to Android device:

```bash
adb push deeplab_v3/dlv3_qnn.pte /data/local/tmp/dlv3_qnn.pte
```

### Runtime

We build the required ExecuTorch runtime library (AAR) to run the model.

#### XNNPACK

```bash
# go to ExecuTorch repo root
export ANDROID_NDK=<path-to-android-ndk>
export ANDROID_ABIS=arm64-v8a

# Run the following lines from the `executorch/` folder
./install_executorch.sh --clean

# Create a new directory `app/libs` for the AAR to live
pushd examples/demo-apps/android/ExecuTorchDemo
mkdir -p app/libs
popd

# Build the AAR. It will include XNNPACK backend by default.
export BUILD_AAR_DIR=$(realpath examples/demo-apps/android/ExecuTorchDemo/app/libs)
sh build/build_android_library.sh
```

#### Qualcomm Hexagon NPU

```bash
# go to ExecuTorch repo root
export ANDROID_NDK=<path-to-android-ndk>
export ANDROID_ABIS=arm64-v8a
export QNN_SDK_ROOT=<path-to-qnn-sdk-root>

# Run the following lines from the `executorch/` folder
./install_executorch.sh --clean

# Create a new directory `app/libs` for the AAR to live
pushd examples/demo-apps/android/ExecuTorchDemo
mkdir -p app/libs
popd

# Build the AAR. It will include XNNPACK backend by default.
export BUILD_AAR_DIR=$(realpath examples/demo-apps/android/ExecuTorchDemo/app/libs)
sh build/build_android_library.sh
```

This is very similar to XNNPACK setup, but users now needs to define `QNN_SDK_ROOT` so that
QNN backend is built into the AAR.

## Running the App

1. Open the project `examples/demo-apps/android/ExecuTorchDemo` with Android Studio.

2. [Run](https://developer.android.com/studio/run) the app (^R).

<img src="_static/img/android_studio.png" alt="Android Studio View" /><br>

On the phone or emulator, you can try running the model:
<img src="_static/img/android_demo_run.png" alt="Android Demo" /><br>

## Takeaways
Through this tutorial we've learnt how to build the ExecuTorch runtime library with XNNPACK (or Qualcomm HTP) backend, and expose it to JNI layer to build the Android app running segmentation model.

## Reporting Issues

If you encountered any bugs or issues following this tutorial please file a bug/issue here on [Github](https://github.com/pytorch/executorch/issues/new).
