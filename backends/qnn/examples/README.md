# executorch QNN Backend examples

This directory contains examples for some AI models.

Please check helper of each examples for detailed arguments.

Here are some general information and limitations.

## Prerequisite

Please finish tutorial [Setting up executorch](../../../docs/website/docs/tutorials/00_setting_up_executorch.md).

Please finish [setup QNN backend](../setup.md).

## Environment

Please set up `QNN_SDK_ROOT` environment variable.
Note that this version should be exactly same as building QNN backend.
Please check [setup](../setup.md).

Please set up `LD_LIBRARY_PATH` to `$QNN_SDK_ROOT/lib/x86_64-linux-clang`.
Or, you could put QNN libraries to default search path of the dynamic linker.

## Device

Please connect an Android phone to the workstation. We use `adb` to communicate with the device.

If the device is in a remote host, you might want to add `-H` to the `adb`
commands in the `SimpleADB` class inside [utils.py](utils.py).


## Please use python xxx.py --help for information of each examples.

Some CLI examples here. Please adjust according to your environment:

```bash
python mobilenet_v2.py -s <device_serial> -m "SM8550" -b path/to/build_android/ -d /path/to/imagenet-mini/val

python deeplab_v3.py -s <device_serial> -m "SM8550" -b path/to/build_android/ --download
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

