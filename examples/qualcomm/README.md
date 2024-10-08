# ExecuTorch QNN Backend examples

This directory contains examples for some AI models.

We have seperated the example scripts into the following subfolders, please refer to [README.md](../../backends/qualcomm/README.md) for the example scripts' directory structure:

1. executor_runner: This folder contains a general executor runner capable of running most of the models. As a rule of thumb, if a model does not have its own customized runner, execute the model using [executor_runner](./executor_runner/qnn_executor_runner.cpp). On the other hand, if a model has its own runner, such as [llama2](./oss_scripts/llama2/qnn_llama_runner.cpp), use the customized runner to execute the model. Customized runner should be located under the same folder as the model's python script.

2. oss_scripts: OSS stands for Open Source Software. This folder contains python scripts for open source models. Some models under this folder might also have their own customized runner.
   For example, [llama2](./oss_scripts/llama2/qnn_llama_runner.cpp) contains not only the python scripts to prepare the model but also a customized runner for executing the model.

3. qaihub_scripts: QAIHub stands for [Qualcomm AI Hub](https://aihub.qualcomm.com/). On QAIHub, users can find pre-compiled context binaries, a format used by QNN to save its models. This provides users with a new option for model deployment. Different from oss_scripts & scripts, which the example scripts are converting a model from nn.Module to ExecuTorch .pte files, qaihub_scripts provides example scripts for converting pre-compiled context binaries to ExecuTorch .pte files. Additionaly, users can find customized example runners specific to the QAIHub models for execution. For example [qaihub_llama2_7b](./qaihub_scripts/llama2/qaihub_llama2_7b.py) is a script converting context binaries to ExecuTorch .pte files, and [qaihub_llama2_7b_runner](./qaihub_scripts/llama2/qaihub_llama2_7b_runner.cpp) is a customized example runner to execute llama2 .pte files. Please be aware that context-binaries downloaded from QAIHub are tied to a specific QNN SDK version.
Before executing the scripts and runner, please ensure that you are using the QNN SDK version that is matching the context binary. Tutorial below will also cover how to check the QNN Version for a context binary.

4. scripts: This folder contains scripts to build models provided by executorch.



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
python mobilenet_v2.py -s <device_serial> -m "SM8550" -b path/to/build-android/ -d /path/to/imagenet-mini/val
```

#### For DeepLab_v3
```bash
python deeplab_v3.py -s <device_serial> -m "SM8550" -b path/to/build-android/ --download
```

#### Check context binary version
```bash
cd ${QNN_SDK_ROOT}/bin/x86_64-linux-clang
./qnn-context-binary-utility --context_binary ${PATH_TO_CONTEXT_BINARY} --json_file ${OUTPUT_JSON_NAME}
```
After retreiving the json file, search in the json file for the field "buildId" and ensure it matches the ${QNN_SDK_ROOT} you are using for the environment variable.
If you run into the following error, that means the ${QNN_SDK_ROOT} that you are using is older than the context binary QNN SDK version. In this case, please download a newer QNN SDK version.
```
Error: Failed to get context binary info.
```

## Additional Dependency

The mobilebert multi-class text classification example requires `pandas` and `sklearn`.
Please install them by something like

```bash
pip install scikit-learn pandas
```

## Limitation

1. QNN 2.24 is used for all examples. Newer or older QNN might work,
but the performance and accuracy number can differ.

2. The mobilebert example is on QNN HTP fp16, which is only supported by a limited
set of SoCs. Please check QNN documents for details.

3. The mobilebert example needs to train the last classifier layer a bit, so it takes
time to run.

4. [**Important**] Due to the numerical limits of FP16, other use cases leveraging mobileBert wouldn't
guarantee to work.
