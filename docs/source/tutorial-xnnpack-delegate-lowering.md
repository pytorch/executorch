# XNNPACK Delegate Lowering Tutorial

The following tutorial will familiarize you with leveraging the ExecuTorch XNNPACK Delegate for accelerating your ML Models using CPU hardware. It will go over exporting and serializing a model to a binary file, targeting the XNNPACK Delegate Backend and running the model on a supported target  platform. To get started quickly, use the script in the ExecuTorch repository with instructions on exporting and generating  a binary file for a few sample models demonstrating the flow.

<!----This will show a grid card on the page----->
::::{grid} 2
:::{grid-item-card}  What you will learn in this tutorial:
:class-card: card-learn
In this tutorial, you will learn how to export an XNNPACK Lowered Model and run it on a target platform
:::
:::{grid-item-card}  Before you begin it is recommended you go through the following:
:class-card: card-prerequisites
* [Installing Buck2](./getting-started-setup.md)
* [Setting up ExecuTorch](./examples-end-to-end-to-lower-model-to-delegate.md)
* [Model Lowering Tutorial](./runtime-backend-delegate-implementation-and-linking.md)
* [Custom Quantization](./quantization-custom-quantization.md) (optional)
* [Executorch XNNPACK Delegate](./native-delegates-XNNPACK-Delegate.md) (optional)
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


edge.exported_program = to_backend(edge.exported_program, XnnpackPartitioner)
```

We will go through this example with the MobileNetV2 model. The flow of lowering a model starts after exporting the model `to_edge`. We call the `to_backend` api with the `XnnpackPartitioner`. The partitioner identifies the subgraphs suitable for XNNPACK lowering. After which these subgraphs, will be serialized with the XNNPACK Delegate flatbuffer schema and will be replaced with calls to the XNNPACK Delegate.

```
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

We print the graph after lowering above to show the new calls to the XNNPACK Delegate, and the subgraphs which were delegated to XNNPACK. The majority of convolution-relu-add blocks and linear blocks were able to be delegated to xnnpack. We can also see the operators which were not able to be lowered to the XNNPACK delegate.

```python
from executorch.examples.export.utils import save_pte_program

exec_prog = edge.to_executorch()
save_pte_program(exec_prog.buffer, "xnnpack_mobilenetv2.pte")
```
After lowering to the XNNPACK Program, we can then prepare it for executorch and save the model as a pte file.


## Lowering a Quantized model to XNNPACK
The XNNPACK Delegate is also a backend for executing symmetrically quantized models. Understanding quantization flow and how to quantize models can be read about [Custom Quantization](quantization-custom-quantization.md), but for the sake of this tutorial we will leverage the quantize function conveniently added to the executorch/examples folder.

```python
import torch._export as export
from executorch.examples.xnnpack.quantization.utils import quantize
from executorch.exir import EdgeCompileConfig

mobilenet_v2 = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

mobilenet_v2 = export.capture_pre_autograd_graph(mobilenet_v2, sample_inputs)
quantized_mobilenetv2 = quantize(mobilenet_v2, sample_inputs)
```

Quantization requires two stage export, so we first use the `capture_pre_autograd_graph` api to capture the model before giving it to quantize. After performing these quantization steps, we can now leverage the XNNPACK Delegate to lower quantized subgraphs, and use XNNPACK as a backend for executing quantized models. We can now follow the same steps for lowering to XNNPACK to now lower this quantized model.

```python
edge = export_to_edge(
    mobilenet_v2,
    example_inputs,
    edge_compile_config=EdgeCompileConfig(_check_ir_validity=False)
)
edge.exported_program = to_backend(edge.exported_program, XnnpackPartitioner)

exec_prog = edge.to_executorch()
save_pte_program(exec_prog.buffer, "qs8_xnnpack_mobilenetv2.pte")
```

## Lowering with aot_compiler.py script
We have also provided a script to quickly lower and export a few example models. You can run the script to generate lowered fp32 and quantized models

```
python3 -m examples.xnnpack.aot_compiler.py --model_name="mv2" --quantize --delegate
```

Note in above, —model_name specifies the model to use, the —quantize flag controls whether the model is quantized, the —delegate flag controls whether we lower to the xnnpack delegate. The generated model file will be named [model_name]_xnnpack_[qs8/fp32].pte

## Running the XNNPACK model on your host platform
We will use buck2 to run the XNNPACK Lowered model on your host platform. You can follow the instructions here to install [buck2](getting-started-setup.md). Once you have your lowered model, you can run with the prebuilt `xnn_executor_runner` provided in the examples. This will simply run the model on some sample inputs.

```
buck2 run examples/backend:xnn_executor_runner -- --model_path ./mv2_xnnpack_fp32.pte
```

## Building and Linking with the XNNPACK Backend
You can build the XNNPACK Backend [target](https://github.com/pytorch/executorch/blob/main/backends/xnnpack/targets.bzl#L54) and link it with your binary. For more information on Backend Delegate linking you can take a look at this [resource](runtime-backend-delegate-implementation-and-linking.md)

After building the XNNPACK Backend and exporting your XNNPACK Delegated Model, you can also integrate the model ad backend into your app to leverage the CPU acceleration on your device. You can follow the following guide on how to integrate the XNNPACK into your app (TBD)
