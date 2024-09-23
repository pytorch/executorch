# ExecuTorch XNNPACK Delegate

This subtree contains the XNNPACK Delegate implementation for ExecuTorch.
XNNPACK is an optimized library of neural network inference operators for ARM
and x86 CPUs. It is an open source project used by PyTorch. The delegate is the
mechanism for leveraging the XNNPACK library to accelerate operators running on
CPU.

## Layout
- `cmake/` : CMake related files
- `operators`: the directory to store all of op visitors
    - `node_visitor.py`: Implementation of serializing each lowerable operator
      node
    - ...
- `partition/`: Partitioner is used to identify operators in model's graph that
  are suitable for lowering to XNNPACK delegate
    - `xnnpack_partitioner.py`: Contains partitioner that tags graph patterns
      for XNNPACK lowering
    - `configs.py`: Contains lists of op/modules for XNNPACK lowering
- `passes/`: Contains passes which are used before preprocessing to prepare the
  graph for XNNPACK lowering
- `runtime/` : Runtime logic used at inference. This contains all the cpp files
  used to build the runtime graph and execute the XNNPACK model
- `serialization/`: Contains files related to serializing the XNNPACK graph
  representation of the PyTorch model
    - `schema.fbs`: Flatbuffer schema of serialization format
    - `xnnpack_graph_schema.py`: Python dataclasses mirroring the flatbuffer
      schema
    - `xnnpack_graph_serialize`: Implementation for serializing dataclasses
      from graph schema to flatbuffer
- `test/`: Tests for XNNPACK Delegate
- `third-party/`: third-party libraries used by XNNPACK Delegate
- `xnnpack_preprocess.py`: Contains preprocess implementation which is called
  by `to_backend` on the graph or subgraph of a model returning a preprocessed
  blob responsible for executing the graph or subgraph at runtime

## End to End Example

To further understand the features of the XNNPACK Delegate and how to use it, consider the following end to end example with MobilenetV2.

### Lowering a model to XNNPACK
```python
import torch
import torchvision.models as models

from torch.export import export, ExportedProgram
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeProgramManager, ExecutorchProgramManager, to_edge
from executorch.exir.backend.backend_api import to_backend


mobilenet_v2 = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

exported_program: ExportedProgram = export(mobilenet_v2, sample_inputs)
edge: EdgeProgramManager = to_edge(exported_program)

edge = edge.to_backend(XnnpackPartitioner())
```

We will go through this example with the [MobileNetV2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/) pretrained model downloaded from the TorchVision library. The flow of lowering a model starts after exporting the model `to_edge`. We call the `to_backend` api with the `XnnpackPartitioner`. The partitioner identifies the subgraphs suitable for XNNPACK backend delegate to consume. Afterwards, the identified subgraphs will be serialized with the XNNPACK Delegate flatbuffer schema and each subgraph will be replaced with a call to the XNNPACK Delegate.

```python
>>> print(edge.exported_program().graph_module)
GraphModule(
  (lowered_module_0): LoweredBackendModule()
  (lowered_module_1): LoweredBackendModule()
)

def forward(self, arg314_1):
    lowered_module_0 = self.lowered_module_0
    executorch_call_delegate = torch.ops.higher_order.executorch_call_delegate(lowered_module_0, arg314_1);  lowered_module_0 = arg314_1 = None
    getitem = executorch_call_delegate[0];  executorch_call_delegate = None
    aten_view_copy_default = executorch_exir_dialects_edge__ops_aten_view_copy_default(getitem, [1, 1280]);  getitem = None
    aten_clone_default = executorch_exir_dialects_edge__ops_aten_clone_default(aten_view_copy_default);  aten_view_copy_default = None
    lowered_module_1 = self.lowered_module_1
    executorch_call_delegate_1 = torch.ops.higher_order.executorch_call_delegate(lowered_module_1, aten_clone_default);  lowered_module_1 = aten_clone_default = None
    getitem_1 = executorch_call_delegate_1[0];  executorch_call_delegate_1 = None
    return (getitem_1,)
```

We print the graph after lowering above to show the new nodes that were inserted to call the XNNPACK Delegate. The subgraphs which are being delegated to XNNPACK are the first argument at each call site. It can be observed that the majority of `convolution-relu-add` blocks and `linear` blocks were able to be delegated to XNNPACK. We can also see the operators which were not able to be lowered to the XNNPACK delegate, such as `clone` and `view_copy`.

```python
exec_prog = edge.to_executorch()

with open("xnnpack_mobilenetv2.pte", "wb") as file:
    exec_prog.write_to_file(file)
```
After lowering to the XNNPACK Program, we can then prepare it for executorch and save the model as a `.pte` file. `.pte` is a binary format that stores the serialized ExecuTorch graph.


### Running the XNNPACK Model with CMake
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
```

## Help & Improvements
If you have problems or questions, or have suggestions for ways to make
implementation and testing better, please reach out to the PyTorch Edge team or
create an issue on [github](https://www.github.com/pytorch/executorch/issues).


## See Also
For more information about the XNNPACK Delegate, please check out the following resources:
- [ExecuTorch XNNPACK Delegate](https://pytorch.org/executorch/0.2/native-delegates-executorch-xnnpack-delegate.html)
- [Building and Running ExecuTorch with XNNPACK Backend](https://pytorch.org/executorch/0.2/native-delegates-executorch-xnnpack-delegate.html)
