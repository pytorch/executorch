# Building and Running ExecuTorch with XNNPACK Backend

The following tutorial will familiarize you with leveraging the ExecuTorch XNNPACK Delegate for accelerating your ML Models using CPU hardware. It will go over exporting and serializing a model to a binary file, targeting the XNNPACK Delegate Backend and running the model on a supported target  platform. To get started quickly, use the script in the ExecuTorch repository with instructions on exporting and generating  a binary file for a few sample models demonstrating the flow.

<!----This will show a grid card on the page----->
::::{grid} 2
:::{grid-item-card}  What you will learn in this tutorial:
:class-card: card-prerequisites
In this tutorial, you will learn how to export an XNNPACK lowered Model and run it on a target platform
:::
:::{grid-item-card}  Before you begin it is recommended you go through the following:
:class-card: card-prerequisites
* [Setting up ExecuTorch](./getting-started-setup.md)
* [Model Lowering Tutorial](./tutorials/export-to-executorch-tutorial)
* [ExecuTorch XNNPACK Delegate](./native-delegates-executorch-xnnpack-delegate.md)
:::
::::


## Lowering a model to XNNPACK
```python
import torch
import torchvision.models as models

from torch.export import export, ExportedProgram
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeProgramManager, ExecutorchProgramManager, to_edge_transform_and_lower
from executorch.exir.backend.backend_api import to_backend


mobilenet_v2 = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

exported_program: ExportedProgram = export(mobilenet_v2, sample_inputs)
edge: EdgeProgramManager = to_edge_transform_and_lower(
    exported_program,
    partitioner=[XnnpackPartitioner()],
)
```

We will go through this example with the [MobileNetV2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/) pretrained model downloaded from the TorchVision library. The flow of lowering a model starts after exporting the model `to_edge`. We call the `to_backend` api with the `XnnpackPartitioner`. The partitioner identifies the subgraphs suitable for XNNPACK backend delegate to consume. Afterwards, the identified subgraphs will be serialized with the XNNPACK Delegate flatbuffer schema and each subgraph will be replaced with a call to the XNNPACK Delegate.

```python
>>> print(edge.exported_program().graph_module)
GraphModule(
  (lowered_module_0): LoweredBackendModule()
  (lowered_module_1): LoweredBackendModule()
)



def forward(self, b_features_0_1_num_batches_tracked, ..., x):
    lowered_module_0 = self.lowered_module_0
    lowered_module_1 = self.lowered_module_1
    executorch_call_delegate_1 = torch.ops.higher_order.executorch_call_delegate(lowered_module_1, x);  lowered_module_1 = x = None
    getitem_53 = executorch_call_delegate_1[0];  executorch_call_delegate_1 = None
    aten_view_copy_default = executorch_exir_dialects_edge__ops_aten_view_copy_default(getitem_53, [1, 1280]);  getitem_53 = None
    aten_clone_default = executorch_exir_dialects_edge__ops_aten_clone_default(aten_view_copy_default);  aten_view_copy_default = None
    executorch_call_delegate = torch.ops.higher_order.executorch_call_delegate(lowered_module_0, aten_clone_default);  lowered_module_0 = aten_clone_default = None
    getitem_52 = executorch_call_delegate[0];  executorch_call_delegate = None
    return (getitem_52,)
```

We print the graph after lowering above to show the new nodes that were inserted to call the XNNPACK Delegate. The subgraphs which are being delegated to XNNPACK are the first argument at each call site. It can be observed that the majority of `convolution-relu-add` blocks and `linear` blocks were able to be delegated to XNNPACK. We can also see the operators which were not able to be lowered to the XNNPACK delegate, such as `clone` and `view_copy`.

```python
exec_prog = edge.to_executorch()

with open("xnnpack_mobilenetv2.pte", "wb") as file:
    exec_prog.write_to_file(file)
```
After lowering to the XNNPACK Program, we can then prepare it for executorch and save the model as a `.pte` file. `.pte` is a binary format that stores the serialized ExecuTorch graph.


## Lowering a Quantized Model to XNNPACK
The XNNPACK delegate can also execute symmetrically quantized models. To understand the quantization flow and learn how to quantize models, refer to [Custom Quantization](quantization-custom-quantization.md) note. For the sake of this tutorial, we will leverage the `quantize()` python helper function conveniently added to the `executorch/executorch/examples` folder.

```python
from torch.export import export_for_training
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

mobilenet_v2 = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

mobilenet_v2 = export_for_training(mobilenet_v2, sample_inputs).module() # 2-stage export for quantization path

from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)


def quantize(model, example_inputs):
    """This is the official recommended flow for quantization in pytorch 2.0 export"""
    print(f"Original model: {model}")
    quantizer = XNNPACKQuantizer()
    # if we set is_per_channel to True, we also need to add out_variant of quantize_per_channel/dequantize_per_channel
    operator_config = get_symmetric_quantization_config(is_per_channel=False)
    quantizer.set_global(operator_config)
    m = prepare_pt2e(model, quantizer)
    # calibration
    m(*example_inputs)
    m = convert_pt2e(m)
    print(f"Quantized model: {m}")
    # make sure we can export to flat buffer
    return m

quantized_mobilenetv2 = quantize(mobilenet_v2, sample_inputs)
```

Quantization requires a two stage export. First we use the `export_for_training` API to capture the model before giving it to `quantize` utility function. After performing the quantization step, we can now leverage the XNNPACK delegate to lower the quantized exported model graph. From here, the procedure is the same as for the non-quantized model lowering to XNNPACK.

```python
# Continued from earlier...
edge = to_edge_transform_and_lower(
    export(quantized_mobilenetv2, sample_inputs),
    compile_config=EdgeCompileConfig(_check_ir_validity=False),
    partitioner=[XnnpackPartitioner()]
)

exec_prog = edge.to_executorch()

with open("qs8_xnnpack_mobilenetv2.pte", "wb") as file:
    exec_prog.write_to_file(file)
```

## Lowering with `aot_compiler.py` script
We have also provided a script to quickly lower and export a few example models. You can run the script to generate lowered fp32 and quantized models. This script is used simply for convenience and performs all the same steps as those listed in the previous two sections.

```
python -m examples.xnnpack.aot_compiler --model_name="mv2" --quantize --delegate
```

Note in the example above,
* the `-—model_name` specifies the model to use
* the `-—quantize` flag controls whether the model should be quantized or not
* the `-—delegate` flag controls whether we attempt to lower parts of the graph to the XNNPACK delegate.

The generated model file will be named `[model_name]_xnnpack_[qs8/fp32].pte` depending on the arguments supplied.

## Running the XNNPACK Model with CMake
After exporting the XNNPACK Delegated model, we can now try running it with example inputs using CMake. We can build and use the xnn_executor_runner, which is a sample wrapper for the ExecuTorch Runtime and XNNPACK Backend. We first begin by configuring the CMake build like such:
```bash
# cd to the root of executorch repo
cd executorch

# Get a clean cmake-out directory
rm -rf cmake-out
mkdir cmake-out

# Configure cmake
cmake \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -DPYTHON_EXECUTABLE=python \
    -Bcmake-out .
```
Then you can build the runtime componenets with

```bash
cmake --build cmake-out -j9 --target install --config Release
```

Now you should be able to find the executable built at `./cmake-out/backends/xnnpack/xnn_executor_runner` you can run the executable with the model you generated as such
```bash
./cmake-out/backends/xnnpack/xnn_executor_runner --model_path=./mv2_xnnpack_fp32.pte
# or to run the quantized variant
./cmake-out/backends/xnnpack/xnn_executor_runner --model_path=./mv2_xnnpack_q8.pte
```

## Building and Linking with the XNNPACK Backend
You can build the XNNPACK backend [CMake target](https://github.com/pytorch/executorch/blob/main/backends/xnnpack/CMakeLists.txt#L83), and link it with your application binary such as an Android or iOS application. For more information on this you may take a look at this [resource](demo-apps-android.md) next.
