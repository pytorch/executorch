# ExecuTorch QNN Backend examples

This directory contains examples for some AI models.

Please check helper of each examples for detailed arguments.

Here are some general information and limitations.

## Prerequisite

Please finish tutorial [Setting up executorch](https://pytorch.org/executorch/stable/getting-started-setup).

Please finish [setup QNN backend](../../docs/source/build-run-qualcomm-ai-engine-direct-backend.md).

## Environment

Please set up `QNN_SDK_ROOT` environment variable.
Note that this version should be exactly same as building QNN backend.
Please check [setup](../../docs/source/build-run-qualcomm-ai-engine-direct-backend.md).

Please set up `LD_LIBRARY_PATH` to `$QNN_SDK_ROOT/lib/x86_64-linux-clang`.
Or, you could put QNN libraries to default search path of the dynamic linker.

## Device

Please connect an Android phone to the workstation. We use `adb` to communicate with the device.

If the device is in a remote host, you might want to add `-H` to the `adb`
commands in the `SimpleADB` class inside [utils.py](utils.py).

## Please use python xxx.py --help for information of each examples.

Some CLI examples here. Please adjust according to your environment:

#### First switch to following folder
```bash
cd $EXECUTORCH_ROOT/examples/qualcomm/scripts
```

#### For MobileNet_v2
```bash
python mobilenet_v2.py -s <device_serial> -m "SM8550" -b path/to/cmake-out-android/ -d /path/to/imagenet-mini/val
```

#### For DeepLab_v3
```bash
python deeplab_v3.py -s <device_serial> -m "SM8550" -b path/to/cmake-out-android/ --download
```

## Additional Dependency

The mobilebert multi-class text classification example requires `pandas` and `sklearn`.
Please install them by something like

```bash
pip install scikit-learn pandas
```

## Limitation

1. QNN 2.13 is used for all examples. Newer or older QNN might work,
but the performance and accuracy number can differ.

2. The mobilebert example is on QNN HTP fp16, which is only supported by a limited
set of SoCs. Please check QNN documents for details.

3. The mobilebert example needs to train the last classifier layer a bit, so it takes
time to run.

4. [**Important**] Due to the numerical limits of FP16, other use cases leveraging mobileBert wouldn't
guarantee to work.
