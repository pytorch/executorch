# Exporting custom LLMs

If you have your own PyTorch model that is an LLM, this guide shows how to
manually export and lower it to ExecuTorch. Use this flow when your model is not
covered by the native [`export_llm`](export-llm.md) API, is not directly handled
by [Optimum ExecuTorch](export-llm-optimum.md), or needs model-specific changes
before it can use the standard ExecuTorch LLM runtime.

This example uses Karpathy’s [nanoGPT](https://github.com/karpathy/nanoGPT), which is a minimal implementation of
GPT-2 124M. The same manual export pattern applies broadly to PyTorch models.
However, exporting a `.pte` file and running that file with the stock LLM runners
are separate steps. To use the LLM runners, the exported model must also follow
the runtime contract described below.


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

## Using the LLM runners with a custom model

The exported `.pte` file can be loaded directly through the ExecuTorch runtime,
but many text-generation applications use the higher-level LLM runners described
in [Running LLMs with C++](run-with-c-plus-plus.md). These runners handle
tokenization, prefill, decode, sampling, and streaming output. To use them with a
custom model, shape the model boundary around autoregressive text generation:

A KV cache stores previous attention key and value tensors so decode can append
new tokens without recomputing attention over the full context.

- The model should accept token IDs as the primary input.
- If the model uses a KV cache, it should also accept a position input, often
  named `input_pos` or `start_pos`.
- The model should return a single logits tensor that the runner can sample from.
- The tokenizer file and BOS/EOS token IDs should match the model.
- Cache tensors should normally be model-owned buffers, not extra inputs and
  outputs passed through the runner.

A typical runner-compatible forward signature looks like:

```python
def forward(
    self,
    tokens: torch.Tensor,
    input_pos: torch.Tensor,
) -> torch.Tensor:
    ...
    return logits
```

Models without a KV cache may expose only the token input, but generation will
usually be much slower because the model recomputes attention over the full
context on each decode step.

The runner also reads metadata from the `.pte` file. At minimum, include the
values that describe sequence limits, KV cache behavior, and tokenizer
termination:

- `get_max_seq_len`: maximum number of tokens processed by one model invocation.
- `get_max_context_len`: maximum context length remembered by the model.
- `use_kv_cache`: whether the model has an internal KV cache.
- `enable_dynamic_shape`: whether prefill can use dynamic sequence lengths.
- `get_bos_id` and `get_eos_ids`: token IDs used by the runner.

For example:

```python
metadata = {
    "get_bos_id": bos_id,
    "get_eos_ids": [eos_id],
    "get_max_seq_len": max_seq_len,
    "get_max_context_len": max_context_len,
    "use_kv_cache": True,
    "enable_dynamic_shape": True,
}
```

When manually exporting, serialize this metadata as constant methods. Constant
methods are named values in the `.pte` file that the runner can query at load
time:

```python
edge_manager = to_edge(
    traced_model,
    constant_methods=metadata,
    compile_config=edge_config,
)
```

If your model needs additional runtime inputs, such as explicit cache tensors,
attention masks, encoder outputs, or cross-attention state, the default text LLM
runner is probably not the right boundary. In that case, either wrap the model so
that those values are stored inside the module, or build a custom runner or
`IOManager` for the model-specific input and output protocol. An `IOManager` is
the runner component that prepares model inputs and processes model outputs for
prefill and decode.

Encoder-decoder models, such as translation models from Fairseq, are a common
case where this distinction matters. ExecuTorch can run the exported program,
but the stock text-generation runner is oriented around decoder-only generation.
If the model is supported by Optimum ExecuTorch, prefer that path. Otherwise,
decide whether to wrap the model into the runner-compatible shape or expose a
custom runtime interface.

## Adapting attention and KV cache

Optimized LLM exports work well in ExecuTorch when attention and decode state are
structured in an export-friendly way. The important design choice is to keep the
runtime interface simple while moving mutable decode state into the module.

The optimized transformer implementations in ExecuTorch preserve a few
properties that are useful to keep in a custom model:

- The exported graph is static enough for `torch.export`: tensor operations are
  traceable, and generation state does not depend on Python-side mutation.
- The runner boundary stays small: tokens and optional position go in, logits
  come out, and metadata describes how to drive generation.
- KV cache state is stored in model buffers and updated by tensor position.
- Attention is factored so standard scaled dot product attention (SDPA) can be
  replaced by optimized or backend-specific SDPA implementations when the tensor
  layout matches.
- Large compute patterns, such as linear layers and attention, stay recognizable
  to backend partitioners.

For KV cache support:

- Register key and value caches as module buffers so they are part of the
  exported program state.
- Update cache entries using the tensor position passed to the model, rather than
  Python-side counters or data-dependent control flow.
- Keep cache shapes predictable. Backends and custom operators often rely on
  fixed cache layout assumptions.
- Return logits only. The default runner does not expect cache tensors as model
  outputs.
- Reset or reinitialize cache state through the runner/runtime lifecycle, not by
  changing Python attributes during generation.

For attention:

- Prefer standard `torch.nn.functional.scaled_dot_product_attention` or an
  equivalent module boundary that can later be swapped for backend-specific
  attention.
- Keep query, key, value, mask, and cache shapes explicit and stable.
- First make the model exportable and correct, then apply SDPA, cache,
  quantization, and backend transforms for the targets you care about.

ExecuTorch includes optimized SDPA and cache-update custom operators used by the
Llama export flow. You can leverage those paths when your model's attention
layout matches the expected query/key/value/cache conventions. If your attention
layout is different, it is usually better to adapt the module boundary first
than to force the custom operator into an incompatible shape.

## Reusing LLM components

You do not need to copy the Llama implementation to build a custom model. The
`extension/llm` tree contains reusable pieces that are useful when adapting a
model for export:

- [`extension/llm/modules`](https://github.com/pytorch/executorch/tree/main/extension/llm/modules)
  contains export-friendly modules.
- [`KVCache`](https://github.com/pytorch/executorch/blob/main/extension/llm/modules/kv_cache.py)
  provides an export-friendly cache implementation adapted from torchtune.
- [`MultiHeadAttention`](https://github.com/pytorch/executorch/blob/main/extension/llm/modules/attention.py)
  factors SDPA out of the attention module so it can be replaced with optimized
  implementations.
- [`examples/models/llama/source_transformation`](https://github.com/pytorch/executorch/tree/main/examples/models/llama/source_transformation)
  shows how the Llama flow swaps in custom SDPA, custom KV cache, quantized KV
  cache, and backend-specific attention variants.

These components are most useful as building blocks and reference
implementations. Keep your model architecture readable and close to the original
PyTorch version first, then replace individual pieces only when they improve
export compatibility, runner compatibility, or backend performance.

If you are authoring or fine-tuning a transformer model from scratch, also look
at [torchtune](https://github.com/pytorch/torchtune). Several ExecuTorch LLM
modules are adapted from torchtune modules with changes for export and inference.

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

If you also added runner metadata earlier, pass the same metadata through the
`constant_methods` argument in this call so the delegated `.pte` keeps the same
runner-visible values.

For custom LLMs, backend performance depends on how much of the model graph the
backend can recognize and delegate. Keep the following in mind when adapting the
model:

- Inspect delegated and non-delegated operators after lowering with
  `get_delegation_info()`.
- Prefer linear and attention patterns that leave model weights visible as
  constants to the partitioner.
- Be careful with dynamic shapes inside delegated subgraphs. Dynamic prefill can
  be useful, but not every dynamic pattern is backend-friendly.
- Export separate `.pte` files for different targets, such as XNNPACK for CPU
  and Core ML for Apple devices.

When targeting XNNPACK, use the XNNPACK partitioner and quantization flow. For
details, see the [XNNPACK backend overview](../backends/xnnpack/xnnpack-overview.md),
[XNNPACK quantization](../backends/xnnpack/xnnpack-quantization.md), and
[XNNPACK troubleshooting](../backends/xnnpack/xnnpack-troubleshooting.md).

When targeting Core ML, follow the Core ML backend configuration and validate on
the target Apple OS and hardware. Stateful KV cache and fused SDPA support can be
backend- and OS-version dependent. For details, see the
[Core ML backend overview](../backends/coreml/coreml-overview.md),
[Core ML partitioner](../backends/coreml/coreml-partitioner.md), and
[Core ML troubleshooting](../backends/coreml/coreml-troubleshooting.md).

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
