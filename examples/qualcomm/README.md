# ExecuTorch QNN Backend examples

This directory contains examples for some AI models.

We have separated the example scripts into the following subfolders, please refer to [README.md](../../backends/qualcomm/README.md) for the example scripts' directory structure:

1. executor_runner: This folder contains a general executor runner capable of running most of the models. As a rule of thumb, if a model does not have its own customized runner, execute the model using [executor_runner](executor_runner/qnn_executor_runner.cpp). On the other hand, if a model has its own runner, such as [llama](oss_scripts/llama/qnn_llama_runner.cpp), use the customized runner to execute the model. Customized runner should be located under the same folder as the model's python script.

2. oss_scripts: OSS stands for Open Source Software. This folder contains python scripts for open source models. Some models under this folder might also have their own customized runner.
   For example, [llama](oss_scripts/llama/qnn_llama_runner.cpp) contains not only the python scripts to prepare the model but also a customized runner for executing the model.

3. scripts: This folder contains scripts to build models provided by ExecuTorch.

4. util_scripts: This folder includes tutorial example scripts designed to showcase the utilities we've developed. For example, we provide a debugging tool [qnn_intermediate_debugger](./util_scripts/qnn_intermediate_debugger_demo.py) that allow users to compare the intermediate outputs of QNNs V.S. CPUs. By reviewing these scripts, we aim to help users smoothly integrate these utilities into their own projects.



Please check helper of each examples for detailed arguments.

Here are some general information and limitations.

## Prerequisite

Please finish tutorial [Setting up ExecuTorch](https://pytorch.org/executorch/main/getting-started-setup).

Please finish [setup QNN backend](../../docs/source/backends-qualcomm.md).

## Environment

Please set up `QNN_SDK_ROOT` environment variable.
Note that this version should be exactly same as building QNN backend.
Please check [setup](../../docs/source/backends-qualcomm.md).

Please set up `LD_LIBRARY_PATH` to `$QNN_SDK_ROOT/lib/x86_64-linux-clang`.
Or, you could put QNN libraries to default search path of the dynamic linker.

## Device

Please connect an Android phone to the workstation. We use `adb` to communicate with the device.

If the device is in a remote host, you might want to add `-H` to the `adb`
commands in the `SimpleADB` class inside [export_utils.py](../../backends/qualcomm/export_utils.py).

## Please use python xxx.py --help for information of each examples.

Some CLI examples here. Please adjust according to your environment. If you want to export the model without running it, please add `--compile_only` to the command.:

#### First switch to following folder
```bash
cd $EXECUTORCH_ROOT/examples/qualcomm/scripts
```

## Simple Examples to Verify the Backend is Working
```bash
python export_example.py -m add -g
```

It will generate a simple add model targeting for "SM8550". You can manually push the `add.pte` file to the device following https://pytorch.org/executorch/stable/build-run-qualcomm-ai-engine-direct-backend.html and run it with

```bash
./qnn_executor_runner --model_path add.pte
```

#### For MobileNet_v2
```bash
python mobilenet_v2.py -s <device_serial> -m "SM8550" -b path/to/build-android/ -d /path/to/imagenet-mini/val
```

#### For DeepLab_v3
```bash
python deeplab_v3.py -s <device_serial> -m "SM8550" -b path/to/build-android/ --download
```

## Model Structure
This section outlines the essential APIs and utilities provided to streamline the process of model conversion, deployment, and evaluation on Qualcomm hardware using ExecuTorch. The official APIs can be found under [export_utils.py](../../backends/qualcomm/export_utils.py)

1. `setup_common_args_and_variables()`:

   `setup_common_args_and_variables()` returns an `argparse.ArgumentParser`. This parser defines both required and optional arguments, which can later be passed into the ExecuTorch QNN API, `QnnConfig.load_config()`.

2. `QnnConfig.load_config()`:

   `QnnConfig.load_config` accepts either:
      1. An `argparse.ArgumentParser` created by `setup_common_args_and_variables()`
      2. A `.json` configuration file. A sample file is provided under [sample_config.json](./sample_config.json) for reference.

   This function returns a `QnnConfig`, which serves as an input to some of the key APIs that will be covered below: `build_executorch_binary()`, `SimpleADB`.

3. `build_executorch_binary()`:

   `build_executorch_binary` is a high-level API used to convert a PyTorch model into a Qualcomm-compatible .pte binary format. This function streamlines the process of quantization, transformation, optimization, and export, enabling users to efficiently deploy models on Qualcomm hardware.

4. `SimpleADB`:

   `SimpleADB` provides a simplified interface for interacting with Android devices. It allows users to execute ADB commands such as:     
      1. Push necessary artifacts to device
      2. Execute the runner
      3. Pull the execution outputs/results


## Run Inference Using Shared Buffer
This section shows how to use shared buffer for input/output tensors in QNN ExecuTorch, usually graph inputs and outputs on shared memory to reduce huge tensor copying time from CPU to HTP. This feature can accelerate inference speed. Users need to do shared memory resource management by themselves. The key idea is to use `QnnExecuTorchAllocCustomMem` to allocate a large chunk of memory on the device, then use `QnnExecuTorchFreeCustomMem` to free it after inference.

### Run example scipts with shared buffer
You can specify `--shared_buffer` flag to run example scripts with shared buffer such as:
```
python mobilenet_v2.py -s <device_serial> -m "SM8550" -b path/to/build-android/ -d /path/to/imagenet-mini/val --shared_buffer
```

### Workflow of using shared memory
There are two ways to use shared buffer in QNN ExecuTorch:
1. Use ION buffer (1 tensor to 1 rpc mem)
    - For all I/O tensors, user call QnnExecuTorchAllocCustomMem to request n bytes RPC memory
    - For all I/O tensors, user create TensorImpl with the above memory address
    - Run inference with shared buffer
    - For all I/O tensors, user call QnnExecuTorchFreeCustomMem to free RPC memory
2. Use Custom Memory (many tensors to 1 rpc mem)
    - Call QnnExecuTorchAllocCustomMem to allocate a large RPC memory block capable of holding all I/O tensors
    - For all I/O tensors, create TensorImpl with a sufficient memory block derived from the base RPC memory address, then call QnnExecuTorchAddCustomMemTensorAddr to bind each tensor’s address to the base RPC memory.
    - Run inference with shared buffer
    - Call QnnExecuTorchFreeCustomMem to free RPC memory

## Additional Dependency
This example requires the following Python packages:
- pandas and scikit-learn: used in the mobilebert multi-class text classification example.
- graphviz (optional): used for visualizing QNN graphs during debugging.

Please install them by something like
```bash
pip install scikit-learn pandas graphviz
```

## Limitation

1. QNN 2.37 is used for all examples. Newer or older QNN might work, but the performance and accuracy number can differ.

2. The mobilebert example is on QNN HTP fp16, which is only supported by a limited
set of SoCs. Please check QNN documents for details.

3. The mobilebert example needs to train the last classifier layer a bit, so it takes
time to run.

4. [**Important**] Due to the numerical limits of FP16, other use cases leveraging mobileBert wouldn't
guarantee to work.
