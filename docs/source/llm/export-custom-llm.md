# Exporting custom LLMs

If you have your own PyTorch model that is an LLM, this guide will show you how to manually export and lower to ExecuTorch, with many of the same optimizations as covered in the previous `export_llm` guide.

This example uses Karpathy’s [nanoGPT](https://github.com/karpathy/nanoGPT), which is a minimal implementation of
GPT-2 124M. This guide is applicable to other language models, as ExecuTorch is model-invariant.


## Exporting to ExecuTorch (basic)

Exporting takes a PyTorch model and converts it into a format that can run efficiently on consumer devices.

For this example, you will need the nanoGPT model and the corresponding tokenizer vocabulary.

::::{tab-set}
:::{tab-item} curl
```
curl https://raw.githubusercontent.com/karpathy/nanoGPT/master/model.py -O
curl https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json -O
```
:::
:::{tab-item} wget
```
wget https://raw.githubusercontent.com/karpathy/nanoGPT/master/model.py
wget https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json
```
:::
::::

To convert the model into a format optimized for standalone execution, there are two steps. First, use the PyTorch
`export` function to convert the PyTorch model into an intermediate, platform-independent intermediate representation. Then
use the ExecuTorch `to_edge` and `to_executorch` methods to prepare the model for on-device execution. This creates a .pte
file which can be loaded by a desktop or mobile application at runtime.

Create a file called export_nanogpt.py with the following contents:

```python
# export_nanogpt.py

import torch

from executorch.exir import EdgeCompileConfig, to_edge
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.export import export

from model import GPT

# Load the model.
model = GPT.from_pretrained('gpt2')

# Create example inputs. This is used in the export process to provide
# hints on the expected shape of the model input.
example_inputs = (torch.randint(0, 100, (1, model.config.block_size), dtype=torch.long), )

# Set up dynamic shape configuration. This allows the sizes of the input tensors
# to differ from the sizes of the tensors in `example_inputs` during runtime, as
# long as they adhere to the rules specified in the dynamic shape configuration.
# Here we set the range of 0th model input's 1st dimension as
# [0, model.config.block_size].
# See https://pytorch.org/executorch/main/concepts#dynamic-shapes
# for details about creating dynamic shapes.
dynamic_shape = (
    {1: torch.export.Dim("token_dim", max=model.config.block_size)},
)

# Trace the model, converting it to a portable intermediate representation.
# The torch.no_grad() call tells PyTorch to exclude training-specific logic.
with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
    m = export(model, example_inputs, dynamic_shapes=dynamic_shape).module()
    traced_model = export(m, example_inputs, dynamic_shapes=dynamic_shape)

# Convert the model into a runnable ExecuTorch program.
edge_config = EdgeCompileConfig(_check_ir_validity=False)
edge_manager = to_edge(traced_model,  compile_config=edge_config)
et_program = edge_manager.to_executorch()

# Save the ExecuTorch program to a file.
with open("nanogpt.pte", "wb") as file:
    file.write(et_program.buffer)
```

To export, run the script with `python export_nanogpt.py` (or python3, as appropriate for your environment). It will generate a `nanogpt.pte` file in the current directory.

For more information, see [Exporting to ExecuTorch](../tutorials/export-to-executorch-tutorial) <!-- @lint-ignore --> and
[torch.export](https://pytorch.org/docs/stable/export.html).

## Backend delegation

While ExecuTorch provides a portable, cross-platform implementation for all
operators, it also provides specialized backends for a number of different
targets. These include, but are not limited to, x86 and ARM CPU acceleration via
the XNNPACK backend, Apple acceleration via the Core ML backend and Metal
Performance Shader (MPS) backend, and GPU acceleration via the Vulkan backend.

Because optimizations are specific to a given backend, each pte file is specific
to the backend(s) targeted at export. To support multiple devices, such as
XNNPACK acceleration for Android and Core ML for iOS, export a separate PTE file
for each backend.

To delegate a model to a specific backend during export, ExecuTorch uses the
`to_edge_transform_and_lower()` function. This function takes the exported program
from `torch.export` and a backend-specific partitioner object. The partitioner
identifies parts of the computation graph that can be optimized by the target
backend. Within `to_edge_transform_and_lower()`, the exported program is
converted to an edge dialect program. The partitioner then delegates compatible
graph sections to the backend for acceleration and optimization. Any graph parts
not delegated are executed by ExecuTorch's default operator implementations.

To delegate the exported model to a specific backend, we need to import its
partitioner as well as edge compile config from ExecuTorch codebase first, then
call `to_edge_transform_and_lower`.

Here's an example of how to delegate nanoGPT to XNNPACK (if you're deploying to an Android phone for instance):

```python
# export_nanogpt.py

# Load partitioner for Xnnpack backend
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

# Model to be delegated to specific backend should use specific edge compile config
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

import torch
from torch.export import export
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.export import export

from model import GPT

# Load the nanoGPT model.
model = GPT.from_pretrained('gpt2')

# Create example inputs. This is used in the export process to provide
# hints on the expected shape of the model input.
example_inputs = (
        torch.randint(0, 100, (1, model.config.block_size - 1), dtype=torch.long),
    )

# Set up dynamic shape configuration. This allows the sizes of the input tensors
# to differ from the sizes of the tensors in `example_inputs` during runtime, as
# long as they adhere to the rules specified in the dynamic shape configuration.
# Here we set the range of 0th model input's 1st dimension as
# [0, model.config.block_size].
# See ../concepts.html#dynamic-shapes
# for details about creating dynamic shapes.
dynamic_shape = (
    {1: torch.export.Dim("token_dim", max=model.config.block_size - 1)},
)

# Trace the model, converting it to a portable intermediate representation.
# The torch.no_grad() call tells PyTorch to exclude training-specific logic.
with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
    m = export(model, example_inputs, dynamic_shapes=dynamic_shape).module()
    traced_model = export(m, example_inputs, dynamic_shapes=dynamic_shape)

# Convert the model into a runnable ExecuTorch program.
# To be further lowered to Xnnpack backend, `traced_model` needs xnnpack-specific edge compile config
edge_config = get_xnnpack_edge_compile_config()
# Converted to edge program and then delegate exported model to Xnnpack backend
# by invoking `to` function with Xnnpack partitioner.
edge_manager = to_edge_transform_and_lower(traced_model, partitioner = [XnnpackPartitioner()], compile_config = edge_config)
et_program = edge_manager.to_executorch()

# Save the Xnnpack-delegated ExecuTorch program to a file.
with open("nanogpt.pte", "wb") as file:
    file.write(et_program.buffer)
```


## Quantization

Quantization refers to a set of techniques for running calculations and storing tensors using lower precision types.
Compared to 32-bit floating point, using 8-bit integers can provide both a significant speedup and reduction in
memory usage. There are many approaches to quantizing a model, varying in amount of pre-processing required, data
types used, and impact on model accuracy and performance.

Because compute and memory are highly constrained on mobile devices, some form of quantization is necessary to ship
large models on consumer electronics. In particular, large language models, such as Llama2, may require quantizing
model weights to 4 bits or less.

Leveraging quantization requires transforming the model before export. PyTorch provides the pt2e (PyTorch 2 Export)
API for this purpose. This example targets CPU acceleration using the XNNPACK delegate. As such, it needs to use the
 XNNPACK-specific quantizer. Targeting a different backend will require use of the corresponding quantizer.

To use 8-bit integer dynamic quantization with the XNNPACK delegate, call `prepare_pt2e`, calibrate the model by
running with a representative input, and then call `convert_pt2e`. This updates the computational graph to use
quantized operators where available.

```python
# export_nanogpt.py

from executorch.backends.transforms.duplicate_dynamic_quant_chain import (
    DuplicateDynamicQuantChainPass,
)
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
```

```python
# Use dynamic, per-channel quantization.
xnnpack_quant_config = get_symmetric_quantization_config(
    is_per_channel=True, is_dynamic=True
)
xnnpack_quantizer = XNNPACKQuantizer()
xnnpack_quantizer.set_global(xnnpack_quant_config)

m = export(model, example_inputs).module()

# Annotate the model for quantization. This prepares the model for calibration.
m = prepare_pt2e(m, xnnpack_quantizer)

# Calibrate the model using representative inputs. This allows the quantization
# logic to determine the expected range of values in each tensor.
m(*example_inputs)

# Perform the actual quantization.
m = convert_pt2e(m, fold_quantize=False)
DuplicateDynamicQuantChainPass()(m)

traced_model = export(m, example_inputs)
```

Additionally, add or update the `to_edge_transform_and_lower()` call to use `XnnpackPartitioner`. This
instructs ExecuTorch to optimize the model for CPU execution via the XNNPACK backend.

```python
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
```

```python
edge_config = get_xnnpack_edge_compile_config()
# Convert to edge dialect and lower to XNNPack.
edge_manager = to_edge_transform_and_lower(traced_model, partitioner = [XnnpackPartitioner()], compile_config = edge_config)
et_program = edge_manager.to_executorch()

with open("nanogpt.pte", "wb") as file:
    file.write(et_program.buffer)
```

For more information, see [Quantization in ExecuTorch](../quantization-overview.md).

## Profiling and Debugging
After lowering a model by calling `to_edge_transform_and_lower()`, you may want to see what got delegated and what didn’t. ExecuTorch
provides utility methods to give insight on the delegation. You can use this information to gain visibility into
the underlying computation and diagnose potential performance issues. Model authors can use this information to
structure the model in a way that is compatible with the target backend.

### Visualizing the Delegation

The `get_delegation_info()` method provides a summary of what happened to the model after the `to_edge_transform_and_lower()` call:

```python
from executorch.devtools.backend_debug import get_delegation_info
from tabulate import tabulate

# ... After call to to_edge_transform_and_lower(), but before to_executorch()
graph_module = edge_manager.exported_program().graph_module
delegation_info = get_delegation_info(graph_module)
print(delegation_info.get_summary())
df = delegation_info.get_operator_delegation_dataframe()
print(tabulate(df, headers="keys", tablefmt="fancy_grid"))
```

For nanoGPT targeting the XNNPACK backend, you might see the following (note that the numbers below are for illustration purposes only and actual values may vary):
```
Total  delegated  subgraphs:  145
Number  of  delegated  nodes:  350
Number  of  non-delegated  nodes:  760
```


|    |  op_type                                 |  # in_delegated_graphs  |  # in_non_delegated_graphs  |
|----|---------------------------------|------- |-----|
|  0  |  aten__softmax_default  |  12  |  0  |
|  1  |  aten_add_tensor  |  37  |  0  |
|  2  |  aten_addmm_default  |  48  |  0  |
|  3  |  aten_any_dim  |  0  |  12  |
|      |  ...  |    |    |
|  25  |  aten_view_copy_default  |  96  |  122  |
|      |  ...  |    |    |
|  30  |  Total  |  350  |  760  |

From the table, the operator `aten_view_copy_default` appears 96 times in delegate graphs and 122 times in non-delegated graphs.
To see a more detailed view, use the `format_delegated_graph()` method to get a formatted str of printout of the whole graph or use `print_delegated_graph()` to print directly:

```python
from executorch.exir.backend.utils import format_delegated_graph
graph_module = edge_manager.exported_program().graph_module
print(format_delegated_graph(graph_module))
```
This may generate a large amount of output for large models. Consider using "Control+F" or "Command+F" to locate the operator you’re interested in
(e.g. “aten_view_copy_default”). Observe which instances are not under lowered graphs.

In the fragment of the output for nanoGPT below, observe that a transformer module has been delegated to XNNPACK while the where operator is not.

```
%aten_where_self_22 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.where.self](args = (%aten_logical_not_default_33, %scalar_tensor_23, %scalar_tensor_22), kwargs = {})
%lowered_module_144 : [num_users=1] = get_attr[target=lowered_module_144]
backend_id: XnnpackBackend
lowered graph():
    %p_transformer_h_0_attn_c_attn_weight : [num_users=1] = placeholder[target=p_transformer_h_0_attn_c_attn_weight]
    %p_transformer_h_0_attn_c_attn_bias : [num_users=1] = placeholder[target=p_transformer_h_0_attn_c_attn_bias]
    %getitem : [num_users=1] = placeholder[target=getitem]
    %sym_size : [num_users=2] = placeholder[target=sym_size]
    %aten_view_copy_default : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.view_copy.default](args = (%getitem, [%sym_size, 768]), kwargs = {})
    %aten_permute_copy_default : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.permute_copy.default](args = (%p_transformer_h_0_attn_c_attn_weight, [1, 0]), kwargs = {})
    %aten_addmm_default : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.addmm.default](args = (%p_transformer_h_0_attn_c_attn_bias, %aten_view_copy_default, %aten_permute_copy_default), kwargs = {})
    %aten_view_copy_default_1 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.view_copy.default](args = (%aten_addmm_default, [1, %sym_size, 2304]), kwargs = {})
    return [aten_view_copy_default_1]
```

### Further Model Analysis and Debugging

Through the [ExecuTorch's Developer Tools](getting-started.md#performance-analysis), users are able to profile model execution, giving timing information for each operator in the model, doing model numeric debugging, etc.

An ETRecord is an artifact generated at the time of export that contains model graphs and source-level metadata linking the ExecuTorch program to the original PyTorch model. You can view all profiling events without an ETRecord, though with an ETRecord, you will also be able to link each event to the types of operators being executed, module hierarchy, and stack traces of the original PyTorch source code. For more information, see [the ETRecord docs](../etrecord.rst).

In your export script, after calling `to_edge()` and `to_executorch()`, call `generate_etrecord()` with the `EdgeProgramManager` from `to_edge()` and the `ExecuTorchProgramManager` from `to_executorch()`. Make sure to copy the `EdgeProgramManager`, as the call to `to_edge_transform_and_lower()` mutates the graph in-place.

```
# export_nanogpt.py

import copy
from executorch.devtools import generate_etrecord

# Make the deep copy immediately after to to_edge()
edge_manager_copy = copy.deepcopy(edge_manager)

# ...
# Generate ETRecord right after to_executorch()
etrecord_path = "etrecord.bin"
generate_etrecord(etrecord_path, edge_manager_copy, et_program)
```

Run the export script and the ETRecord will be generated as `etrecord.bin`.

To learn more about ExecuTorch's Developer Tools, see the [Introduction to the ExecuTorch Developer Tools](../devtools-overview.md).
