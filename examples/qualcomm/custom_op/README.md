# Custom Operator Support
The Qualcomm AI Engine Direct Backend in ExecuTorch supports custom PyTorch operators via the Qualcomm AI Engine Direct Op Package mechanism. Custom PyTorch operators, utilizing the torch.library API, can be successfully delegated and supported through user-written op packages. Additionally, built-in PyTorch nodes can be overridden by these op packages.

Note: The Qualcomm AI Engine Direct SDK is required to compile an OP package.

This folder contains examples demonstrating how to register custom operators into PyTorch and how to register their op packages into the Qualcomm AI Engine Direct Backend in ExecuTorch.
## Prerequisite

- Please finish tutorial [Setting up executorch](https://pytorch.org/executorch/stable/getting-started-setup).

- Please finish [setup QNN backend](../../../docs/source/backends-qualcomm.md).

- Please follow [the instructions to install proper version of Hexagon SDK and Hexagon Tools.](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/linux_setup.html#htp-and-dsp)
  - This example is verified with SM8650 (Snapdragon 8 Gen 3).
  - Install hexagon-sdk-5.4.0, hexagon-sdk-6.0.0, and hexagon tool 8.8.02
  ```bash
  # install hexagon sdk 5.4.0
  qpm-cli --install hexagonsdk5.x --version 5.4.0.3 --path /path/to/Qualcomm/Hexagon_SDK/hexagon-sdk-5.4.0
  # install hexagon sdk 6.0.0
  qpm-cli --install hexagonsdk6.x --version 6.0.0.2 --path /path/to/Qualcomm/Hexagon_SDK/hexagon-sdk-6.0.0
  # install hexagon tool 8.8.02
  qpm-cli --extract hexagon8.8 --version 8.8.02.1 --path /path/to/Qualcomm/Hexagon_SDK/hexagon-sdk-6.0.0/tools/HEXAGON_Tools/8.8.02
  ```

## Setup environment variables
`$HEXAGON_SDK_ROOT` refers to the root of the specified version of Hexagon SDK, i.e., the directory containing `readme.txt`

`$X86_CXX` refers to the clang++ compiler, verified with clang++9

```bash
export HEXAGON_SDK_ROOT=/path/to/Qualcomm/Hexagon_SDK/hexagon-sdk-5.4.0
export X86_CXX=/path/to/clang-9.0.0/bin/clang++
```


## Instructions to build and run the example
Use the following command, we can get the op package for the custom op `ExampleCustomOp`. And then compiling the custom model containing the custom op `torch.ops.my_ops.mul3.default` to Qualcomm AI Engine Direct binary with the op package.

```bash
python3 examples/qualcomm/custom_op/custom_ops_1.py --build_folder build-android -s <device_serial> -H <host> -m SM8650 --op_package_dir examples/qualcomm/custom_op/example_op_package_htp/ExampleOpPackage --build_op_package
```

## How to quantize custom op in Qualcomm AI Engine Direct backend
Use the custom annotation in Qnn Quantizer
```python
quantizer = make_quantizer(
    quant_dtype=quant_dtype, custom_annotations=(annotate_custom,)
)
```

## Generating Op Packages
To generate operation (op) packages, follow these steps:

1. Define an XML OpDef Configuration File:
    - Create an XML file that describes the package information, including the package name, version, and domain.
    - Specify the operations the package contains. Refer to [the example op package XML file](example_op_package_htp/ExampleOpPackage/config/example_op_package_htp.xml) for guidance.
2. Generate Skeleton Sample Code:
    - Once the XML file is fully defined according to the specifications, pass it as an argument to the `qnn-op-package-generator` tool using the --config_path or -p option.
    - This will generate the skeleton sample code.
3. Implement the Operations:
    - The generated interface generally does not require extra implementation.
    - The source files will contain empty function bodies that need to be completed by users. Refer to [the example op package for implementation details](example_op_package_htp/ExampleOpPackage/src/ops/ExampleCustomOp.cpp).
4. Support Custom PyTorch Operators:
    - To support the parameters of custom PyTorch operators, a custom op builder is generated from the meta and `_schema.argument` of `torch.fx.Node`.
    - Ensure that the OpDef of the op package aligns with the schema of the custom PyTorch operators.

## Op package format 
### Inputs 
in[0]…in[m-1]

The same number of input tensors as defined in the PyTorch custom op. Where ``m`` is
the number of inputs.

* Mandatory: true
* Data type: backend specific
* Shape: Any

### Parameters

Optionally, define one or more parameters for the operation.
* Mandatory: true
* Data type: backend specific
* Shape: Any

### Outputs
out[0]

For now, only support one output tensors.

* Mandatory: true
* Data type: backend specific
* Shape: Any

Consult the Qualcomm AI Engine Direct documentation for information on [generation op packages](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/op_def_schema.html).

## Registering Op Packages
After an op package library has been generated, certain information needs to be passed to the `compile_spec` in order to properly delegate the nodes. [The example script](custom_ops_1.py) shows how to construct the `QnnExecuTorchOpPackageOptions` and register op packages with the `compile spec`.
