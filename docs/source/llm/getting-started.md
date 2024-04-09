# Getting Started with LLMs via ExecuTorch

### Table Of Contents


1.  Prerequisites
2.  Hello World Example
3.  Quantization
4.  Using Mobile Acceleration
5.  Debugging and Profiling
6.  How to use custom kernels
7.  How to build mobile apps


## Prerequisites

To follow this guide, you'll need to clone the ExecuTorch repository and install dependencies. 
ExecuTorch requires Python 3.10, as well as  This example uses
conda to manage the Python environment. 

::::{tab-set}
:::{tab-item} conda
Instructions on installing miniconda can be [found here](https://docs.anaconda.com/free/miniconda).

```
# Create a directory for this example.
mkdir et-nanogpt
cd et-nanogpt

# Clone the ExecuTorch repository and submodules.
mkdir third-party
git clone -b release/0.2 https://github.com/pytorch/executorch.git third-party/executorch
cd third-party/executorch
git submodule update --init

# Create a conda environment and install requirements.
conda create -yn executorch python=3.10.0
conda activate executorch
pip install cmake zstd
./install_requirements.sh

cd ../..
```
:::
:::{tab-item} pyenv-virtualenv
Instructions on installing pyenv-virtualenv can be [found here](https://github.com/pyenv/pyenv-virtualenv?tab=readme-ov-file#installing-with-homebrew-for-macos-users).

Importantly, if installing pyenv through brew, it does not automatically enable pyenv in the terminal, leading to errors. Run the following commands to enable.
See the pyenv-virtualenv installation guide above on how to add this to your .bashrc or .zshrc to avoid needing to run these commands manually.
```
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

```
# Create a directory for this example.
mkdir et-nanogpt
cd et-nanogpt

pyenv install 3.10
pyenv virtualenv 3.10 executorch
pyenv activate executorch

# Clone the ExecuTorch repository and submodules.
mkdir third-party
git clone -b release/0.2 https://github.com/pytorch/executorch.git third-party/executorch
cd third-party/executorch
git submodule update --init

# Install requirements.
pip3 install cmake zstd
./install_requirements.sh

cd ../..
```
:::
::::

For more information, see [Setting Up ExecuTorch](https://pytorch.org/executorch/stable/getting-started-setup.html).


## Running a Large Language Model Locally

This example uses Karpathy’s [NanoGPT](https://github.com/karpathy/nanoGPT), which is a minimal implementation of 
GPT-2 124M. This guide is applicable to other language models, as ExecuTorch is model-invariant.

There are two steps to running a model with ExecuTorch:

1.  Export the model. This step preprocesses it into a format suitable for runtime execution.
2.  At runtime, load the model file and run with the ExecuTorch runtime.

<br />

The export step happens ahead of time, typically as part of the application build or when the model changes. The resultant
.pte file is distributed with the application. At runtime, the application loads the .pte file and passes it to the
ExecuTorch runtime.

### Step 1. Exporting to ExecuTorch

Exporting takes a PyTorch model and converts it into a format that can run efficiently on consumer devices.

For this example, you will need the NanoGPT model and the corresponding tokenizer vocabulary.

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

from executorch.exir import EdgeCompileConfig,  to_edge
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch._export import capture_pre_autograd_graph
from torch.export import export

from model import GPT

# Load the model.
model = GPT.from_pretrained('gpt2')

# Create example inputs. This is used in the export process to provide 
# hints on the expected shape of the model input.
example_inputs = (torch.randint(0, 100, (1, 8), dtype=torch.long), )

# Trace the model, converting it to a portable intermediate representation.
# The torch.no_grad() call tells PyTorch to exclude training-specific logic.
with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
    m = capture_pre_autograd_graph(model, example_inputs)
    traced_model = export(m, example_inputs)

# Convert the model into a runnable ExecuTorch program.
edge_config = EdgeCompileConfig(_check_ir_validity=False)
edge_manager = to_edge(traced_model,  compile_config=edge_config)
et_program = edge_manager.to_executorch()

# Save the ExecuTorch program to a file.
with open("nanogpt.pte", "wb") as file:
    file.write(et_program.buffer)
```

To export, run the script with `python export_nanogpt.py` (or python3, as appropriate for your environment). It will generate a `nanogpt.pte` file in the current directory.

For more information, see [Exporting to ExecuTorch](https://pytorch.org/executorch/main/tutorials/export-to-executorch-tutorial.html) and
[torch.export](https://pytorch.org/docs/stable/export.html).

### Step 2. Invoking the Runtime

ExecuTorch provides a set of runtime APIs and types to load and run models.

Create a file called main.cpp with the following contents:

```cpp
// main.cpp

#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>

#include "basic_tokenizer.h"
#include "basic_sampler.h"
#include "managed_tensor.h"

#include <executorch/extension/module/module.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

using namespace torch::executor;

using SizesType = exec_aten::SizesType;
using DimOrderType = exec_aten::DimOrderType;
using StridesType = exec_aten::StridesType;
```

The model inputs and outputs take the form of tensors. A tensor can be thought of as an multi-dimensional array.
The ExecuTorch `EValue` class provides a wrapper around tensors and other ExecuTorch data types.

Since the LLM generates one token at a time, the driver code needs to repeatedly invoke the model, building the
output token by token. Each generated token is passed as input for the next run.

```cpp
// main.cpp

std::string generate(
    Module& llm_model,
    std::string& prompt,
    BasicTokenizer& tokenizer,
    BasicSampler& sampler,
    size_t max_output_length) {

    // Convert the input text into a list of integers (tokens) that represents 
    // it, using the string-to-token mapping that the model was trained on. 
    // Each token is an integer that represents a word or part of a word.
    std::vector<int64_t> input_tokens = tokenizer.encode(prompt);
    std::vector<int64_t> output_tokens;

    for (auto i = 0u; i < max_output_length; i++) {
        // Convert the input_tokens from a vector of int64_t to EValue.
        // EValue is a unified data type in the ExecuTorch runtime.
        ManagedTensor tensor_tokens(
            input_tokens.data(), 
            {1, static_cast<int>(input_tokens.size())}, 
            ScalarType::Long);
        std::vector<EValue> inputs = {tensor_tokens.get_tensor()};

        // Run the model. It will return a tensor of logits (log-probabilities).
        Result<std::vector<EValue>> logits_evalue = llm_model.forward(inputs);

        // Convert the output logits from EValue to std::vector, which is what
        // the sampler expects.
        Tensor logits_tensor = logits_evalue.get()[0].toTensor();
        std::vector<float> logits(logits_tensor.data_ptr<float>(), 
            logits_tensor.data_ptr<float>() + logits_tensor.numel());

        // Sample the next token from the logits.
        int64_t next_token = sampler.sample(logits);
        output_tokens.push_back(next_token);

        std::cout << tokenizer.decode({ next_token });
        std::cout.flush();

        // Update next input.
        input_tokens.erase(input_tokens.begin());
        input_tokens.push_back(next_token);
    }

    std::cout << std::endl;

    // Convert the output tokens into a human-readable string.
    std::string output_string = tokenizer.decode(output_tokens);
    return output_string;
}
```

The `Module` class handles loading the .pte file and preparing for execution. 

The tokenizer is responsible for converting from a human-readable string representation of the prompt to the 
numerical form expected by the model. To do this, the tokenzier associates short substrings with a given token ID.
The tokens can be thought of as representing words or parts of words, though, in-practice, they may be arbitrary
sequences of characters.

The tokenizer loads the vocabulary from a file, which contains the mapping between each token ID and the text it
represents. Call `tokenizer.encode()` and `tokenizer.decode()` to convert between string and token representations.

The sampler is responsible for selecting the next token, based on the logits, or log-probabilties, output by the
model. The LLM returns a logit value for each possible next token. The sampler chooses which token to use based
on some strategy. The simplest approach, used here, is to take the token with the highest logit value.

Samplers may provide configurable options, such as configurable amount of randomness to the outputs selection,
penalties for repeated tokens, and biases to prioritize or de-prioritize specific tokens.


```cpp
// main.cpp

int main() {
    // Set up the prompt. This provides the seed text for the model to elaborate.
    std::string prompt = "Once upon a time, there was a";

    // The tokenizer is used to convert between tokens (used by the model) and
    // human-readable strings.
    BasicTokenizer tokenizer("vocab.json");

    // The sampler is used to sample the next token from the logits.
    BasicSampler sampler = BasicSampler();

    // Load the exported nanoGPT program, which was generated via the previous steps.
    Module model("nanogpt.pte", torch::executor::Module::MlockConfig::UseMlockIgnoreErrors);

    const auto max_output_tokens = 30;
    std::cout << prompt;
    generate(model, prompt, tokenizer, sampler, max_output_tokens);
}
```

Finally, download the following files into the same directory as main.h:

TODO: This is a placeholder.
```
curl -O https://raw.githubusercontent.com/GregoryComer/et-tutorials/quantization/nanogpt/managed_tensor.h
curl -O https://raw.githubusercontent.com/GregoryComer/et-tutorials/quantization/nanogpt/basic_tokenizer.h
curl -O https://raw.githubusercontent.com/GregoryComer/et-tutorials/quantization/nanogpt/basic_sampler.h
```

To learn more, see [Running an ExecuTorch Model in C++](https://pytorch.org/executorch/main/running-a-model-cpp-tutorial.html)
and the [ExecuTorch Runtime API Reference](https://pytorch.org/executorch/main/executorch-runtime-api-reference.html).

### Building and Running

ExecuTorch uses the CMake build system. To compile and link against the ExecuTorch runtime,
include the ExecuTorch project via `add_directory` and link against `executorch` and additional
dependencies. 

Create a file named CMakeLists.txt with the following content:

```
# CMakeLists.txt

cmake_minimum_required(VERSION 3.19)
project(nanogpt_runner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set options for executorch build.
option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER "" ON)
option(EXECUTORCH_BUILD_EXTENSION_MODULE "" ON)
option(EXECUTORCH_BUILD_OPTIMIZED "" ON)
option(EXECUTORCH_BUILD_XNNPACK "" ON)

# Include the executorch subdirectory.
add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party/executorch
    ${CMAKE_BINARY_DIR}/third-party/executorch)

add_executable(nanogpt_runner main.cpp)
target_link_libraries(
    nanogpt_runner
    PRIVATE
    executorch
    extension_module_static # Provides the Module class
    optimized_native_cpu_ops_lib) # Provides baseline cross-platform kernels
```

At this point, the working directory should contain the following files:

- CMakeLists.txt
- main.cpp
- basic_tokenizer.h
- basic_sampler.h
- managed_tensor.h
- export_nanogpt.py
- model.py
- vocab.json
- nanogpt.pte

If all of these are present, you can now build and run:
```bash
(rm -rf cmake-out && mkdir cmake-out && cd cmake-out && cmake ..)
cmake --build cmake-out -j10
./cmake-out/nanogpt_runner
```

You should see something like the following:

```
Once upon a time, there was a man who was a member of the military...
```

At this point, it is likely to run very slowly. This is because ExecuTorch hasn't been told to optimize for
specific hardware (delegation), and because it is doing all of the calculations in 32-bit floating point (no quantization).

## Delegation

While ExecuTorch provides a portable, cross-platform implementation for all operators, it also provides specialized 
backends for a number of different targets. These include, but are not limited to, x86 and ARM CPU acceleration via
the XNNPACK backend, Apple acceleration via the CoreML backend and Metal Performance Shader (MPS) backend, and GPU
acceleration via the Vulkan backend.

Because optimizations are specific to a given backend, each pte file is specific to the backend(s) targeted at
export. To support multiple devices, such as XNNPACK acceleration for Android and CoreML for iOS, export a separate
PTE file for each backend.

To delegate to a backend at export time, ExecuTorch provides the `to_backend()` function, which takes a backend-
specific partitioner object. The partitioner is responsible for finding parts of the computation graph that can
be accelerated by the target backend. Any portions of the computation graph not delegated will be executed by the
portable or optimized ExecuTorch implementations.

To delegate to the XNNPACK backend, call `to_backend` with an instance of `XnnpackPartitioner()`.

```python
# export_nanogpt.py

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
from executorch.exir import EdgeCompileConfig, to_edge

#...

edge_config = edge_config = get_xnnpack_edge_compile_config()
edge_manager = to_edge(traced_model, compile_config=edge_config)

# Delegate to the XNNPACK backend.
edge_manager = edge_manager.to_backend(XnnpackPartitioner())

et_program = edge_manager.to_executorch()

```

For more information, see the ExecuTorch guides for the [XNNPACK Backend](https://pytorch.org/executorch/stable/tutorial-xnnpack-delegate-lowering.html)
and [CoreML Backend](https://pytorch.org/executorch/stable/build-run-coreml.html).

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
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
```

```python
# Use dynamic, per-channel quantization.
xnnpack_quant_config = get_symmetric_quantization_config(
    is_per_channel=True, is_dynamic=True
)
xnnpack_quantizer = XNNPACKQuantizer()
xnnpack_quantizer.set_global(xnnpack_quant_config)

m = capture_pre_autograd_graph(model, example_inputs)

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

Additionally, add or update the `to_backend()` call to use `XnnpackPartitioner`. This instructs ExecuTorch to
optimize the model for CPU execution via the XNNPACK backend.

```python
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
```

```python
edge_manager = to_edge(traced_model, compile_config=edge_config)
edge_manager = edge_manager.to_backend(XnnpackPartitioner()) # Lower to XNNPACK.
et_program = edge_manager.to_executorch()
```

Finally, ensure that the runner links against the `xnnpack_backend` target in CMakeLists.txt.

```
add_executable(nanogpt_runner nanogpt_runner.cpp)
target_link_libraries(
    nanogpt_runner
    PRIVATE
    etdump
    extension_module
    portable_ops_lib
    xnnpack_backend) # Link the XNNPACK backend
```

For more information, see [Quantization in ExecuTorch](https://pytorch.org/executorch/stable/quantization-overview.html).

## Profiling and Debugging
After lowering a model by calling `to_backend()`, you may want to see what got delegated and what didn’t. ExecuTorch
provides utility methods to give insight on the delegation. You can use this information to gain visibility into
the underlying computation and diagnose potential performance issues. Model authors can use this information to
structure the model in a way that is compatible with the target backend.

### Visualizing the Delegation

The `get_delegation_info()` method provides a summary of what happened to the model after the `to_backend()` call:

```python
from executorch.exir.backend.utils import get_delegation_info
from tabulate import tabulate

graph_module = edge_manager.exported_program().graph_module
delegation_info = get_delegation_info(graph_module)
print(delegation_info.get_summary())
df = delegation_info.get_operator_delegation_dataframe()
print(tabulate(df, headers="keys", tablefmt="fancy_grid"))
```

For NanoGPT targeting the XNNPACK backend, you might see the following:
```
Total  delegated  subgraphs:  86
Number  of  delegated  nodes:  473
Number  of  non-delegated  nodes:  430
```


|    |  op_type                                 |  occurrences_in_delegated_graphs  |  occurrences_in_non_delegated_graphs  |
|----|---------------------------------|------- |-----|
|  0  |  aten__softmax_default  |  12  |  0  |
|  1  |  aten_add_tensor  |  37  |  0  |
|  2  |  aten_addmm_default  |  48  |  0  |
|  3  |  aten_arange_start_step  |  0  |  25  |
|      |  ...  |    |    |
|  23  |  aten_view_copy_default  |  170  |  48  |
|      |  ...  |    |    |
|  26  |  Total  |  473  |  430  |

From the table, the operator `aten_view_copy_default` appears 170 times in delegate graphs and 48 times in non-delegated graphs.
To see a more detailed view, use the `print_delegated_graph()` method to display a printout of the whole graph.

```python
from executorch.exir.backend.utils import print_delegated_graph
graph_module = edge_manager.exported_program().graph_module
print(print_delegated_graph(graph_module))
```
This may generate a large amount of output for large models. Consider using "Control+F" or "Command+F" to locate the operator you’re interested in 
(e.g. “aten_view_copy_default”). Observe which instances are not under lowered graphs.

In the fragment of the output for NanoGPT below, observe that embedding and add operators are delegated to XNNPACK while the sub operator is not.

```
%aten_unsqueeze_copy_default_22 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.unsqueeze_copy.default](args = (%aten_arange_start_step_23, -2), kwargs = {})
  %aten_unsqueeze_copy_default_23 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.unsqueeze_copy.default](args = (%aten_arange_start_step_24, -1), kwargs = {})
  %lowered_module_0 : [num_users=1] = get_attr[target=lowered_module_0]
    backend_id: XnnpackBackend
    lowered graph():
      %aten_embedding_default : [num_users=1] = placeholder[target=aten_embedding_default]
      %aten_embedding_default_1 : [num_users=1] = placeholder[target=aten_embedding_default_1]
      %aten_add_tensor : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%aten_embedding_default, %aten_embedding_default_1), kwargs = {})
      return (aten_add_tensor,)
  %executorch_call_delegate : [num_users=1] = call_function[target=torch.ops.higher_order.executorch_call_delegate](args = (%lowered_module_0, %aten_embedding_default, %aten_embedding_default_1), kwargs = {})
  %aten_sub_tensor : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.sub.Tensor](args = (%aten_unsqueeze_copy_default, %aten_unsqueeze_copy_default_1), kwargs = {})
```

### Performance Analysis

Through the ExecuTorch SDK, users are able to profile model execution, giving timing information for each operator in the model.

#### Prerequisites

##### ETRecord generation (Optional)

An ETRecord is an artifact generated at the time of export that contains model graphs and source-level metadata linking the ExecuTorch program to the original PyTorch model. You can view all profiling events without an ETRecord, though with an ETRecord, you will also be able to link each event to the types of operators being executed, module hierarchy, and stack traces of the original PyTorch source code. For more information, see [https://pytorch.org/executorch/main/sdk-etrecord.html](https://pytorch.org/executorch/main/sdk-etrecord.html)


In your export script, after calling `to_edge()` and `to_executorch()`, call `generate_etrecord()` with the `EdgeProgramManager` from `to_edge()` and the `ExecuTorchProgramManager` from `to_executorch()`. Make sure to copy the `EdgeProgramManager`, as the call to `to_backend()` mutates the graph in-place.

```
import copy

# Make the deep copy immediately after to to_edge()
edge_program_manager_copy = copy.deepcopy(edge_program_manager)

# ...
# Generate ETRecord right after to_executorch()
etrecord_path = "etrecord.bin"
generate_etrecord(etrecord_path, edge_program_manager_copy, et_program_manager)
```

Run the export script and the ETRecord will be generated as `etrecord.bin`.

##### ETDump generation

An ETDump is an artifact generated at runtime containing a trace of the model execution. For more information, see [https://pytorch.org/executorch/main/sdk-etdump.html](https://pytorch.org/executorch/main/sdk-etdump.html)

Include the ETDump header in your code.
```cpp
// main.cpp

#include <executorch/sdk/etdump/etdump_flatcc.h>
```

Create an Instance of the ETDumpGen class and pass it to the Module constructor.
```cpp
std::unique_ptr<torch::executor::ETDumpGen> etdump_gen_ = std::make_unique<torch::executor::ETDumpGen>();
Module llm_model("nanogpt.pte", Module::MlockConfig::UseMlock, std::move(etdump_gen_));
```

After execution, save the ETDump to a file. You can capture multiple model runs in a single trace, if desired.
```cpp
torch::executor::ETDumpGen* etdump_gen =
    static_cast<torch::executor::ETDumpGen*>(llm_model.event_tracer());

ET_LOG(Info, "ETDump size: %zu blocks", etdump_gen->get_num_blocks());
etdump_result result = etdump_gen->get_etdump_data();
if (result.buf != nullptr && result.size > 0) {
    // On a device with a file system, users can just write it to a file.
    FILE* f = fopen("etdump.etdp", "w+");
    fwrite((uint8_t*)result.buf, 1, result.size, f);
    fclose(f);
    free(result.buf);
}
```

Compile the ExecuTorch runtime with the `ET_EVENT_TRACER_ENABLED` pre-processor flag to enable events to be traced and logged into ETDump inside the ExecuTorch runtime. Add these to your CMakeLists.txt

```
target_compile_options(executorch PUBLIC -DET_EVENT_TRACER_ENABLED)
target_compile_options(portable_ops_lib PUBLIC -DET_EVENT_TRACER_ENABLED)
```
Run the runner, you will see “etdump.etdp” generated.

#### Analyze with Inspector APIs

Once you’ve collected debug artifacts ETDump (and optionally an ETRecord), you can use the Inspector API to view performance information.

```python
from executorch.sdk import Inspector

inspector = Inspector(etdump_path="etdump.etdp", etrecord="etrecord.bin")
# If you did not generate an ETRecord, then just pass in the ETDump: `inspector = Inspector(etdump_path="etdump.etdp")`

inspector.print_data_tabular()
```
This prints the performance data in a tabular format in “inspector_out.txt”, with each row being a profiling event.

|  |  event_block_name  |  event_name  |  p10  (ms)  |  p50  (ms)  |  p90  (ms)  |  avg  (ms)  |  min  (ms)  |  max  (ms)  |  op_types  |  is_delegated_op  |  delegate_backend_name  |
|---|----------------------|------------------|-----------|---------------|--------------|-------------|-------------|--------------|-------------|---------------------------|----------|
|  0  |  Default  |  Method::init  |  60.502  |  60.502  |  60.502  |  60.502  |  60.502  |  60.502  |  []  |  False  |  |
|  1  |  Default  |  Program::load_method  |  60.5114  |  60.5114  |  60.5114  |  60.5114  |  60.5114  |  60.5114  |  []  |  False  |  |
|  2  |  Execute  |  native_call_arange.start_out  |  0.029583  |  0.029583  |  0.029583  |  0.029583  |  0.029583  |  0.029583  |  []  |  False  |  |
|  3  |  Execute  |  native_call_embedding.out  |  0.022916  |  0.022916  |  0.022916  |  0.022916  |  0.022916  |  0.022916  |  []  |  False  |  |
|  4  |  Execute  |  native_call_embedding.out  |  0.001084  |  0.001084  |  0.001084  |  0.001084  |  0.001084  |  0.001084  |  []  |  False  |  |

To learn more about the Inspector and the rich functionality it provides, see the [Inspector API Reference](https://pytorch.org/executorch/main/sdk-inspector.html).

## Custom Kernels
With the ExecuTorch custom operator APIs, custom operator and kernel authors can easily bring in their kernel into PyTorch/ExecuTorch.

There are three steps to use custom kernels in ExecuTorch:

1.  Write the custom kernel using ExecuTorch types.
2.  Compile and link the custom kernel to both AOT Python environment as well as the runtime binary.
3.  Source-to-source transformation to swap an operator with a custom op.

### Writing a Custom Kernel

Define your custom operator schema for both functional variant (used in AOT compilation) and out variant (used in ExecuTorch runtime). The schema needs to follow PyTorch ATen convention (see [native_functions.yaml](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml)).

```
custom_linear(Tensor weight, Tensor input, Tensor(?) bias) -> Tensor

custom_linear.out(Tensor weight, Tensor input, Tensor(?) bias, *, Tensor(a!) out) -> Tensor(a!)
```

Write your custom kernel according to the schema defined above. Use the `EXECUTORCH_LIBRARY` macro to make the kernel available to the ExecuTorch runtime.

```cpp
// custom_linear.h / custom_linear.cpp
#include <executorch/runtime/kernel/kernel_includes.h>

Tensor& custom_linear_out(const Tensor& weight, const Tensor& input, optional<Tensor> bias, Tensor& out) {
    // calculation
    return out;
}

// Register as myop::custom_linear.out
EXECUTORCH_LIBRARY(myop, "custom_linear.out", custom_linear_out);
```

To make this operator available in PyTorch, you can define a wrapper around the ExecuTorch custom kernel. Note that the ExecuTorch
implementation uses ExecuTorch tensor types, while the PyTorch wrapper uses ATen tensors.

```cpp
// custom_linear_pytorch.cpp

#include "custom_linear.h"
#include <torch/library.h>

at::Tensor custom_linear(const at::Tensor& weight, const at::Tensor& input, std::optional<at::Tensor> bias) {

    // initialize out
    at::Tensor out = at::empty({weight.size(1), input.size(1)});

    // wrap kernel in custom_linear.cpp into ATen kernel
    WRAP_TO_ATEN(custom_linear_out, 3)(weight, input, bias, out);

    return out;
}

// Register the operator with PyTorch.
TORCH_LIBRARY(myop,  m) {
    m.def("custom_linear(Tensor weight, Tensor input, Tensor(?) bias) -> Tensor", custom_linear);
    m.def("custom_linear.out(Tensor weight, Tensor input, Tensor(?) bias, *, Tensor(a!) out) -> Tensor(a!)", WRAP_TO_ATEN(custom_linear_out, 3));
}
```

### Compile and Link the Custom Kernel

To make it available to the ExecuTorch runtime, compile custom_linear.h/cpp into the binary target. You can also build the kernel as a dynamically loaded library (.so or .dylib) and link it as well.

To make it available to PyTorch, package custom_linear.h, custom_linear.cpp and custom_linear_pytorch.cpp into a dynamically loaded library (.so or .dylib) and load it into the python environment.
This is needed to make PyTorch aware of the custom operator at the time of export.

```python
import torch
torch.ops.load_library("libcustom_linear.so")
```

Once loaded, you can use the custom operator in PyTorch code.

For more information, see [PyTorch Custom Operators](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html) and
and [ExecuTorch Kernel Registration](https://pytorch.org/executorch/stable/kernel-library-custom-aten-kernel.html).

### Using a Custom Operator in a Model

The custom operator can explicitly used in the PyTorch model, or you can write a transformation to replace instances of a core operator with the custom variant. For this example, you could find
all instances of `torch.nn.Linear` and replace them with `CustomLinear`.

```python
def  replace_linear_with_custom_linear(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                CustomLinear(child.in_features,  child.out_features, child.bias),
        )
        else:
            replace_linear_with_custom_linear(child)
```

The remaining steps are the same as the normal flow. Now you can run this module in eager mode as well as export to ExecuTorch.

## How to build Mobile Apps
You can execute an LLM using ExecuTorch on iOS and Android.

**For iOS see the [iLLaMA App](https://pytorch.org/executorch/main/llm/llama-demo-ios.html).**

**For Android, see the [Android Sample App](https://pytorch.org/executorch/main/llm/llama-demo-android.html).**
