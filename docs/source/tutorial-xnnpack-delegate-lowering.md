# XNNPACK Backend Delegate Lowering Tutorial

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
* [Custom Quantization](./quantization-custom-quantization.md)
* [ExecuTorch XNNPACK Delegate](./native-delegates-executorch-xnnpack-delegate.md)
:::
::::


## Lowering a model to XNNPACK
```python
import torch
import torchvision.models as models

from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.examples.portable.utils import export_to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner


mobilenet_v2 = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

edge = export_to_edge(mobilenet_v2, example_inputs)


edge = edge.to_backend(XnnpackPartitioner)
```

We will go through this example with the MobileNetV2 pretrained model downloaded from the TorchVision library. The flow of lowering a model starts after exporting the model `to_edge`. We call the `to_backend` api with the `XnnpackPartitioner`. The partitioner identifies the subgraphs suitable for XNNPACK backend delegate to consume. Afterwards, the identified subgraphs will be serialized with the XNNPACK Delegate flatbuffer schema and each subgraph will be replaced with a call to the XNNPACK Delegate.

```python
>>> print(edge.exported_program.graph_module)
class GraphModule(torch.nn.Module):
        def forward(self, arg314_1: f32[1, 3, 224, 224]):
            lowered_module_0 = self.lowered_module_0
            executorch_call_delegate = torch.ops.executorch_call_delegate(lowered_module_0, arg314_1)
            getitem: f32[1, 1280, 1, 1] = executorch_call_delegate[0]

            aten_view_copy_default: f32[1, 1280] = executorch_exir_dialects_edge__ops_aten_view_copy_default(getitem, [1, 1280])

            aten_clone_default: f32[1, 1280] = executorch_exir_dialects_edge__ops_aten_clone_default(aten_view_copy_default)

            lowered_module_1 = self.lowered_module_1
            executorch_call_delegate_1 = torch.ops.executorch_call_delegate(lowered_module_1, aten_clone_default)
            getitem_1: f32[1, 1000] = executorch_call_delegate_1[0]
            return (getitem_1,)
```

We print the graph after lowering above to show the new nodes that were inserted to call the XNNPACK Delegate. The subgraphs which are being delegated to XNNPACK are the first argument at each call site. It can be observed that the majority of `convolution-relu-add` blocks and `linear` blocks were able to be delegated to XNNPACK. We can also see the operators which were not able to be lowered to the XNNPACK delegate, such as `clone` and `view_copy`.

```python
from executorch.examples.portable.utils import save_pte_program

exec_prog = edge.to_executorch()
save_pte_program(exec_prog.buffer, "xnnpack_mobilenetv2.pte")
```
After lowering to the XNNPACK Program, we can then prepare it for executorch and save the model as a `.pte` file. `.pte` is a binary format that stores the serialized ExecuTorch graph.


## Lowering a Quantized Model to XNNPACK
The XNNPACK delegate can also execute symmetrically quantized models. To understand the quantization flow and learn how to quantize models, refer to [Custom Quantization](quantization-custom-quantization.md) note. For the sake of this tutorial, we will leverage the `quantize()` python helper function conveniently added to the `executorch/executorch/examples` folder.

```python
import torch._export as export
from executorch.examples.xnnpack.quantization.utils import quantize
from executorch.exir import EdgeCompileConfig

mobilenet_v2 = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

mobilenet_v2 = export.capture_pre_autograd_graph(mobilenet_v2, sample_inputs) # 2-stage export for quantization path
quantized_mobilenetv2 = quantize(mobilenet_v2, sample_inputs)
```

Quantization requires a two stage export. First we use the `capture_pre_autograd_graph` API to capture the model before giving it to `quantize` utility function. After performing the quantization step, we can now leverage the XNNPACK delegate to lower the quantized exported model graph. From here, the procedure is the same as for the non-quantized model lowering to XNNPACK.

```python
# Continued from earlier...
edge = export_to_edge(
    quantized_mobilenetv2,
    example_inputs,
    edge_compile_config=EdgeCompileConfig(_check_ir_validity=False)
)
edge = edge.to_backend(XnnpackPartitioner)

exec_prog = edge.to_executorch()
save_pte_program(exec_prog.buffer, "qs8_xnnpack_mobilenetv2.pte")
```

## Lowering with `aot_compiler.py` script
We have also provided a script to quickly lower and export a few example models. You can run the script to generate lowered fp32 and quantized models. This script is used simply for convenience and performs all the same steps as those listed in the previous two sections.

```
python3 -m examples.xnnpack.aot_compiler.py --model_name="mv2" --quantize --delegate
```

Note in the example above,
* the `-—model_name` specifies the model to use
* the `-—quantize` flag controls whether the model should be quantized or not
* the `-—delegate` flag controls whether we attempt to lower parts of the graph to the XNNPACK delegate.

The generated model file will be named `[model_name]_xnnpack_[qs8/fp32].pte` depending on the arguments supplied.

## Running the XNNPACK Model
We will use `buck2` to run the `.pte` file with XNNPACK delegate instructions in it on your host platform. You can follow the instructions here to install [buck2](getting-started-setup.md). You can now run it with the prebuilt `xnn_executor_runner` provided in the examples. This will run the model on some sample inputs.

```bash
buck2 run examples/backend:xnn_executor_runner -- --model_path ./mv2_xnnpack_fp32.pte
# or to run the quantized variant
buck2 run examples/backend:xnn_executor_runner -- --model_path ./mv2_xnnpack_qs8.pte
```

## Building and Linking with the XNNPACK Backend
You can build the XNNPACK backend [target](https://github.com/pytorch/executorch/blob/main/backends/xnnpack/targets.bzl#L54), and link it with your application binary such as an Android or iOS application. For more information on this you may take a look at this [resource](demo-apps-android.md) next.
