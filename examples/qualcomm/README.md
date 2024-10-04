# ExecuTorch QNN Backend examples

This directory contains examples for some AI models.

We have separated the example scripts into the following subfolders, please refer to [README.md](../../backends/qualcomm/README.md) for the example scripts' directory structure:

1. executor_runner: This folder contains a general executor runner capable of running most of the models. As a rule of thumb, if a model does not have its own customized runner, execute the model using [executor_runner](executor_runner/qnn_executor_runner.cpp). On the other hand, if a model has its own runner, such as [llama](oss_scripts/llama/qnn_llama_runner.cpp), use the customized runner to execute the model. Customized runner should be located under the same folder as the model's python script.

2. oss_scripts: OSS stands for Open Source Software. This folder contains python scripts for open source models. Some models under this folder might also have their own customized runner.
   For example, [llama](oss_scripts/llama/qnn_llama_runner.cpp) contains not only the python scripts to prepare the model but also a customized runner for executing the model.

3. qaihub_scripts: QAIHub stands for [Qualcomm AI Hub](https://aihub.qualcomm.com/). On QAIHub, users can find pre-compiled context binaries, a format used by QNN to save its models. This provides users with a new option for model deployment. Different from oss_scripts & scripts, which the example scripts are converting a model from nn.Module to ExecuTorch .pte files, qaihub_scripts provides example scripts for converting pre-compiled context binaries to ExecuTorch .pte files. Additionally, users can find customized example runners specific to the QAIHub models for execution. For example [qaihub_llama2_7b](qaihub_scripts/llama/llama2/qaihub_llama2_7b.py) is a script converting context binaries to ExecuTorch .pte files, and [qaihub_llama2_7b_runner](qaihub_scripts/llama/llama2/qaihub_llama2_7b_runner.cpp) is a customized example runner to execute llama2 .pte files. Please be aware that context-binaries downloaded from QAIHub are tied to a specific QNN SDK version.
Before executing the scripts and runner, please ensure that you are using the QNN SDK version that is matching the context binary. Please refer to [Check context binary version](#check-context-binary-version) for tutorial on how to check the QNN Version for a context binary.

4. scripts: This folder contains scripts to build models provided by ExecuTorch.

5. util_scripts: This folder includes tutorial example scripts designed to showcase the utilities we've developed. For example, we provide a debugging tool [qnn_intermediate_debugger](./util_scripts/qnn_intermediate_debugger_demo.py) that allow users to compare the intermediate outputs of QNNs V.S. CPUs. By reviewing these scripts, we aim to help users smoothly integrate these utilities into their own projects.



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
commands in the `SimpleADB` class inside [utils.py](utils.py).

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

#### Check context binary version
This is typically useful when users want to run any models under `qaihub_scripts`. When users retrieve context binaries from Qualcomm AI Hub, we need to ensure the QNN SDK used to run the `qaihub_scripts` is the same version as the QNN SDK that Qualcomm AI Hub used to compile the context binaries. To do so, please run the following script to retrieve the JSON file that contains the metadata about the context binary:
```bash
cd ${QNN_SDK_ROOT}/bin/x86_64-linux-clang
./qnn-context-binary-utility --context_binary ${PATH_TO_CONTEXT_BINARY} --json_file ${OUTPUT_JSON_NAME}
```
After retrieving the json file, search in the json file for the field "buildId" and ensure it matches the `${QNN_SDK_ROOT}` you are using for the environment variable.
If you run into the following error, that means the ${QNN_SDK_ROOT} that you are using is older than the context binary's QNN SDK version. In this case, please download a newer QNN SDK version.
```
Error: Failed to get context binary info.
```
## Model Structure
This section outlines the essential APIs and utilities provided to streamline the process of model conversion, deployment, and evaluation on Qualcomm hardware using ExecuTorch.

1. `build_executorch_binary()`:

   build_executorch_binary is a high-level API used to convert a PyTorch model into a Qualcomm-compatible .pte binary format. This function streamlines the process of quantization, transformation, optimization, and export, enabling users to efficiently deploy models on Qualcomm hardware.

2. `SimpleADB`:

   SimpleADB is a Python class that provides a simplified interface for interacting with Android devices. It allows users to execute ADB commands, retrieve device information, and manage files on the device.

3. `get_imagenet_dataset`:
   
   If the model requires ImageNet, this function can be used to load the dataset and apply the necessary preprocessing steps to prepare it for inference or quantization calibration.

4. `topk_accuracy`:

   Calculates the Top-K accuracy for classification models, used to evaluate model performance.

5. `parse_skip_delegation_node`:

   Parses command-line arguments to identify node IDs or operation types that should be skipped during model conversion.

6. `make_output_dir`:

   Creates a clean directory for storing model outputs or intermediate results. If the directory already exists, it will be deleted and recreated to ensure a consistent environment for each run.

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

1. QNN 2.28 is used for all examples. Newer or older QNN might work,
but the performance and accuracy number can differ.

2. The mobilebert example is on QNN HTP fp16, which is only supported by a limited
set of SoCs. Please check QNN documents for details.

3. The mobilebert example needs to train the last classifier layer a bit, so it takes
time to run.

4. [**Important**] Due to the numerical limits of FP16, other use cases leveraging mobileBert wouldn't
guarantee to work.
