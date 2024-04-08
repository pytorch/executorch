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

Let’s start by getting an ExecuTorch environment:

1.  Create a third-party folder (Keeps the file paths organized)
```
mkdir  third-party
cd  third-party
```
2. If you’re new to ExecuTorch follow [these steps](https://pytorch.org/executorch/main/getting-started-setup.html#set-up-your-environment) to set up your environment.

## Instantiating and Executing an LLM

We will use Karpathy’s [NanoGPT](https://github.com/karpathy/nanoGPT) but you can use another model if you prefer.



There are just 2 steps to this:

1.  Export the LLM Model
2.  Create a runtime to execute the model




Note: Reminder to exit out of the “third-party” directory, before proceeding.

### Step 1. Export

[Exporting to ExecuTorch](https://pytorch.org/executorch/main/export-overview.html) simply describes taking an existing model and converting it to the ExecuTorch format.



To start, let’s retrieve our model:

`wget  https://raw.githubusercontent.com/karpathy/nanoGPT/master/model.py`

Next, we’ll create a script (call it export.py) to generate the ExecuTorch Program (which gets dumped into an ExecuTorch Binary):



1.  Create the model and example inputs
```
import torch
from model import GPT

model  =  GPT.from_pretrained('gpt2')
example_inputs = (torch.randint(0, 100, (1, 8), dtype=torch.long), )
```



2.  Trace the model
Tracing extracts a cleaner representation of our model for conversion to ExecuTorch.
You can read more about tracing in [torch.export — PyTorch 2.2 documentation](https://pytorch.org/docs/stable/export.html).

```
from torch.nn.attention import sdpa_kernel,  SDPBackend
from torch._export import capture_pre_autograd_graph
from torch.export import export

# Using a custom SDPA kernel for LLMs
with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]),  torch.no_grad():

m  =  capture_pre_autograd_graph(model,  example_inputs)

traced_model  =  export(m,  example_inputs)
```

3.  Export the model to ExecuTorch
Exporting (or lowering) takes the model and creates a runnable ExecuTorch program, without delegate to any specific bakends for further acceleration.
```
from executorch.exir import EdgeCompileConfig,  to_edge

edge_config  =  EdgeCompileConfig(_check_ir_validity=False)
edge_manager  =  to_edge(traced_model,  compile_config=edge_config)
et_program  =  edge_manager.to_executorch()
```

Also ExecuTorch provides different backend support for mobile acceleration. Simply call `to_backend()` with the specific backend partitioner on edge_manager  during exportation. Take Xnnpack delegation as an example:


```
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
from executorch.exir import EdgeCompileConfig, to_edge

edge_config = edge_config = get_xnnpack_edge_compile_config()
edge_manager = to_edge(traced_model, compile_config=edge_config)
edge_manager = edge_manager.to_backend(XnnpackPartitioner())

et_program = edge_manager.to_executorch()
```

After that, we’re ready to run our model. Remember to save you model before proceeding:

```
#Write the serialized ExecuTorch program to a file.
with open("nanogpt.pte",  "wb") as file:
file.write(et_program.buffer)
```


Then run the script.
`python export.py`

### Step 2. Running the model
Running model stands for executing the exported model on ExecuTorch runtime platform.

Before running, we need to retrieve vocabulary file GPT2 used for tokenization:

```
wget  https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json
```
1.  Create the prompt:
Prompt here means the initial cue given to the model, which it uses as a starting point to generate following sentences. Here we use “Hello world!” as example:


```
string  prompt  =  "Hello world!";
```


2.  Load tokenizer and model
A Tokenizer is a crucial component among different Natural Language Processing (NLP) tasks. The primary functionalities are:

-   Encode: Convert text into structural and numerical representations by parsing text into smaller units.Each unit is replaced by a specific number for the NLP model to consume

-   Decode: Convert the numerical representations back for human interpretation.


In our NanoGPT example, we create a simple tokenizer called BasicTokenizer to demonstrate the function. You can use other implementations like [tiktoken](https://github.com/openai/tiktoken) or your own implementation to do that.


```
#include  "basic_tokenizer.h"
BasicTokenizer tokenizer("vocab.json");
```


To load the exported ExecuTorch model into runtime environment, we can use **Module** class:


```
#include <executorch/extension/module/module.h>
Module llm_model("nanogpt.pte");
```


3.  Tokenize the prompt
```
vector<int64_t> tokens = tokenizer.encode(prompt);
```

4.  Generate outputs
We use the loaded model to generate text based on tokenized prompt. Here we create a helper function to illustrate the pipeline:

```
vector<int64_t> generate(Module& llm_model, vector<int64_t>& input_tokens, BasicSampler& sampler, size_t target_output_length) {
    vector<int64_t> output_tokens;
    for (int i = 0; i < target_output_length; i++) {
        // Convert the input_tokens from a vector of int64_t to EValue.
        // Evalue is a unified data type in the executorch runtime.
        ManagedTensor tensor_tokens(input_tokens.data(), {1, 8}, ScalarType::Long);
        vector<EValue> inputs = {tensor_tokens.get_tensor()};
        // Run the model given the Evalue inputs. The model will also return a sequence of EValues as output.
        Result<vector<EValue>> logits_evalue = llm_model.forward(inputs);
        // Convert the output from EValue to a logits in float.
        Tensor logits_tensor = logits_evalue.get()[0].toTensor();
        vector<float> logits(logits_tensor.data_ptr<float>(), logits_tensor.data_ptr<float>() + logits_tensor.numel());
        // Sample the next token from the logits.
        int64_t next_token = sampler.sample(logits);
        // Record the next token
        output_tokens.push_back(next_token);
        // Update next input.
        input_tokens.erase(input_tokens.begin());
        input_tokens.push_back(next_token);
    }
    return output_tokens;
}

```


And in the main function, we leverage the function to generate the outputs.
```
vector<int64_t> outputs = generate(llm_model, tokens, sampler, /*target_output_length*/20);
```
Notice that here outputs are tokens, rather than actual natural language.

5.  Decode the output.
We convert the generated output tokens back to natural language for better understanding:

```
string out_str = tokenizer.decode(outputs);
```

6.  Print the generated text
```
cout << "output: " << out_str << endl;
```
### Build and Run

1. Create the Cmake file for build
```
cmake_minimum_required(VERSION 3.19)
project(nanogpt_runner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# Set options for executorch build.
option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER "" ON)
option(EXECUTORCH_BUILD_EXTENSION_MODULE "" ON)
option(EXECUTORCH_BUILD_XNNPACK "" ON)
option(EXECUTORCH_BUILD_SDK "" ON) # Needed for etdump

# Include the executorch subdirectory.
add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../executorch
    ${CMAKE_BINARY_DIR}/executorch)

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(nanogpt_runner nanogpt_runner.cpp)
target_link_libraries(
    nanogpt_runner
    PRIVATE
    etdump
    extension_module
    portable_ops_lib)

```

This CMake file links the ExecuTorch codebase, along with the necessary extensions and XNNPACK modules, to the nanogpt runner.

2. Build the c++ environment for nanorunner
```
(rm -rf cmake-out \
  && mkdir cmake-out \
  && cd cmake-out \
  && cmake ..)
```

3. With this CMake file as well as built environment iin place, you can build the nanogpt runner binary by executing the following command:

```
cmake --build cmake-out --target nanogpt_runner -j9
```

4. After the build is complete, you can run the binary with this command:
```
./cmake-out/nanogpt_runner
```
If everything worked it should see something like this:
```
prompt: Hello world!
output: Hello world!

I'm not sure if you've heard of the "Curse of the Dragon" or
```

## Quantization (Optional)

Quantization refers to a set of techniques for running calculations and storing tensors using lower precision types. Compared to 32-bit floating point, using 8-bit integers can provide both a significant speedup and reduction in memory usage. There are many approaches to quantizing a model, varying in amount of pre-processing required, data types used, and impact on model accuracy and performance.

Because compute and memory are highly constrained on mobile devices, some form of quantization is necessary to ship large models on consumer electronics. In particular, large language models, such as Llama2, may require quantizing model weights to 4 bits or less.

Leveraging quantization requires transforming the model before export. PyTorch provides multiple quantization flows. Because we are quantizing a model for export, we need to use the PyTorch 2.0 export (pt2e) quantization API.

This example targets CPU acceleration using the XNNPACK delegate. As such, we need to use the XNNPACK-specific quantizer. Targeting a different backend will require use of the corresponding quantizer.

To use 8-bit integer dynamic quantization with the XNNPACK delegate, perform the following calls prior to calling export. This will update and annotate the computational graph to use quantized operators, where available.

```
from executorch.backends.transforms.duplicate_dynamic_quant_chain import (
    DuplicateDynamicQuantChainPass,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

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

Additionally, add or update the to_backend() call to use XnnpackDynamicallyQuantizedPartitioner. This will instruct the lowering logic to emit the correct quantized operators.

```
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackDynamicallyQuantizedPartitioner,
)

edge_manager = to_edge(traced_model, compile_config=edge_config)

# Lower to XNNPACK using the appropriate quantized partitioner.
edge_manager = edge_manager.to_backend(XnnpackDynamicallyQuantizedPartitioner())

et_program = edge_manager.to_executorch()
```
Finally, update the CMakeLists.txt to link the XNNPACK backend with the runner.

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

## Debugging and Profiling
After lowering a model by calling to_backend(), you might want to see what got delegated and what didn’t. We provide util functions to help you get insight on the delegation, and with such information, you can debug and maybe improve the delegation.

### Debug the Delegation

1.  Get high level information
get_delegation_info gives you a summary of what happened to the model after the to_backend() call:

```
from executorch.exir.backend.utils import get_delegation_info
from tabulate import tabulate

graph_module = edge_manager.exported_program().graph_module
delegation_info = get_delegation_info(graph_module)
print(delegation_info.get_summary())
df = delegation_info.get_operator_delegation_dataframe()
print(tabulate(df, headers="keys", tablefmt="fancy_grid"))
```


Take NanoGPT lowered to XNNPACK as an example:
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
|  4  |  aten_bmm_default  |  24  |  0  |
|  5  |  aten_clone_default  |  0  |  38  |
|  6  |  aten_embedding_default  |  0  |  2  |
|  7  |  aten_expand_copy_default  |  48  |  0  |
|  8  |  aten_full_default  |  0  |  12  |
|  9  |  aten_full_like_default  |  0  |  12  |
|  10  |  aten_gelu_default  |  0  |  12  |
|  11  |  aten_index_tensor  |  0  |  1  |
|  12  |  aten_le_scalar  |  0  |  12  |
|  13  |  aten_logical_and_default  |  0  |  12  |
|  14  |  aten_logical_not_default  |  0  |  12  |
|  15  |  aten_mm_default  |  1  |  0  |
|  16  |  aten_mul_scalar  |  24  |  0  |
|  17  |  aten_native_layer_norm_default  |  0  |  25  |
|  18  |  aten_permute_copy_default  |  109  |  0  |
|  19  |  aten_scalar_tensor_default  |  0  |  12  |
|  20  |  aten_split_with_sizes_copy_default  |  0  |  12  |
|  21  |  aten_sub_tensor  |  0  |  12  |
|  22  |  aten_unsqueeze_copy_default  |  0  |  24  |
|  23  |  aten_view_copy_default  |  170  |  48  |
|  24  |  aten_where_self  |  0  |  12  |
|  25  |  getitem  |  0  |  147  |
|  26  |  Total  |  473  |  430  |

In the table, we see that op type aten_view_copy_default appears 170 times in delegate graphs and 48 times in non-delegated graphs.

| 23 | aten_view_copy_default | 170 | 48 |

From here, we might want to know in which part of the graph it wasn’t delegated. For that, you can use the `print_delegated_graph` util function to see a printout of the whole graph with highlighted lowered graphs.

2.  Print graph module
Call this function right after you call `to_backend()`

```
from executorch.exir.backend.utils import print_delegated_graph
graph_module = self.edge_manager.exported_program().graph_module
print(print_delegated_graph(graph_module))
```

On the printed graph, you can do "Control+F" (or "Command+F" on a Mac) on the operator type you’re interested in (e.g. “aten_view_copy_default”) and observe which ones of them are not under “lowered graph()”s.

### Performance Analysis (Optional)

Through the ExecuTorch SDK, users are able to profile a model and inspect its latency performance.

#### Prerequisites

##### ETRecord generation (Optional)

ETRecord contains model graphs and metadata for linking runtime results (such as profiling) to the eager model. You will be able to view all profiling events with just ETDump (see next section), but with ETRecord, you will also be able to link each event to the types of operators being executed, module hierarchy, and stack traces of the original PyTorch source code. For more information, see [https://pytorch.org/executorch/main/sdk-etrecord.html](https://pytorch.org/executorch/main/sdk-etrecord.html)



**Steps for enablement:**
ETRecord is created during export. In your export script, you just called `to_edge() `and it returned edge_program_manager

```
import copy

# Make the deep copy right after your call to to_edge()
edge_program_manager_copy  =  copy.deepcopy(edge_program_manager)

# ...
# Then generate ETRecord right after your call to to_executorch()
etrecord_path  =  "etrecord.bin"
generate_etrecord(etrecord_path,  edge_program_manager_copy,  et_program_manager)
```
Run the export script, then the ETRecord should be generated under path ./etrecord.bin.

##### ETDump generation

ETDump contains runtime results from executing an ExecuTorch model. For more information, see [https://pytorch.org/executorch/main/sdk-etdump.html](https://pytorch.org/executorch/main/sdk-etdump.html)



**Steps for enablement:**
You need to enable ETDump generation in your nanogpt_runner.cpp.

1.  Include the ETDump header in your code.
```
#include  <executorch/sdk/etdump/etdump_flatcc.h>
```

2.  Create an Instance of the ETDumpGen class and pass it into the Module constructor
```
std::unique_ptr<torch::executor::ETDumpGen> etdump_gen_ = std::make_unique<torch::executor::ETDumpGen>();
Module llm_model("nanogpt.pte", Module::MlockConfig::UseMlock, std::move(etdump_gen_));
```

3.  Dump out the ETDump buffer after call to generate()
```
torch::executor::ETDumpGen* etdump_gen =
static_cast<torch::executor::ETDumpGen*>(llm_model.event_tracer());

ET_LOG(Info, "ETDump size: %zu blocks", etdump_gen->get_num_blocks());
etdump_result result = etdump_gen->get_etdump_data();
if (result.buf != nullptr && result.size > 0) {
// On a device with a file system users can just write it out
// to the file-system.
FILE* f = fopen("etdump.etdp", "w+");
fwrite((uint8_t*)result.buf, 1, result.size, f);
fclose(f);
free(result.buf);
}
```

4.  Compile your binary with the `ET_EVENT_TRACER_ENABLED` pre-processor flag to enable events to be traced and logged into ETDump inside the ExecuTorch runtime. Add these to your CMakeLists.txt

```
target_compile_options(executorch PUBLIC -DET_EVENT_TRACER_ENABLED)
target_compile_options(portable_ops_lib PUBLIC -DET_EVENT_TRACER_ENABLED)
```
Run the runner, you will see “etdump.etdp” generated.

#### Analyze with Inspector APIs

Once you’ve collected debug artifacts ETDump (and the optional ETRecord), you can feed them into Inspector APIs in order to get performance details.

##### Creating an Inspector
```
from executorch.sdk import Inspector

inspector = Inspector(etdump_path="etdump.etdp", etrecord="etrecord.bin")
# If you did not generate an ETRecord, then just pass in ETDump: `inspector = Inspector(etdump_path="etdump.etdp")`
```

Using an Inspector
```
with  open("inspector_out.txt", "w") as file:
    inspector.print_data_tabular(file)
```
This saves the performance data in a tabular format in “inspector_out.txt”, with each row being a profiling event. Top rows:

|  |  event_block_name  |  event_name  |  p10  (ms)  |  p50  (ms)  |  p90  (ms)  |  avg  (ms)  |  min  (ms)  |  max  (ms)  |  op_types  |  is_delegated_op  |  delegate_backend_name  |
|---|----------------------|------------------|-----------|---------------|--------------|-------------|-------------|--------------|-------------|---------------------------|----------|
|  0  |  Default  |  Method::init  |  60.502  |  60.502  |  60.502  |  60.502  |  60.502  |  60.502  |  []  |  False  |  |
|  1  |  Default  |  Program::load_method  |  60.5114  |  60.5114  |  60.5114  |  60.5114  |  60.5114  |  60.5114  |  []  |  False  |  |
|  2  |  Execute  |  native_call_arange.start_out  |  0.029583  |  0.029583  |  0.029583  |  0.029583  |  0.029583  |  0.029583  |  []  |  False  |  |
|  3  |  Execute  |  native_call_embedding.out  |  0.022916  |  0.022916  |  0.022916  |  0.022916  |  0.022916  |  0.022916  |  []  |  False  |  |
|  4  |  Execute  |  native_call_embedding.out  |  0.001084  |  0.001084  |  0.001084  |  0.001084  |  0.001084  |  0.001084  |  []  |  False  |  |

For more information about Inspector APIs and the rich functionality it provides, see [https://pytorch.org/executorch/main/sdk-inspector.html](https://pytorch.org/executorch/main/sdk-inspector.html).

## How to use custom kernels
With our new custom op APIs, custom op/kernel authors can easily bring in their op/kernel into PyTorch/ExecuTorch and the process is streamlined.

There are three steps to use custom kernels in ExecuTorch:

1.  Prepare the kernel implementation using ExecuTorch types.
2.  Compile and link the custom kernel to both AOT Python environment as well as the runner binary.
3.  Source-to-source transformation to swap an operator with a custom op.

### Prepare custom kernel implementation

Define your custom operator schema for both functional variant (used in AOT compilation) and out variant (used in ExecuTorch runtime). The schema needs to follow PyTorch ATen convention (see [native_functions.yaml](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml)). For example:

```
custom_linear(Tensor weight, Tensor input, Tensor(?) bias) -> Tensor

custom_linear.out(Tensor weight, Tensor input, Tensor(?) bias, *, Tensor(a!) out) -> Tensor(a!)
```

Then write your custom kernel according to the schema using ExecuTorch types, along with APIs to register to ExecuTorch runtime:
```
// custom_linear.h/custom_linear.cpp
#include <executorch/runtime/kernel/kernel_includes.h>

Tensor& custom_linear_out(const Tensor& weight, const Tensor& input, optional<Tensor> bias, Tensor& out) {

// calculation
return out;
}

// opset namespace myop
EXECUTORCH_LIBRARY(myop, "custom_linear.out", custom_linear_out);
```

Now we need to write some wrapper for this op to show up in PyTorch, but don’t worry we don’t need to rewrite the kernel. Create a separate .cpp for this purpose:

```
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

// standard API to register ops into PyTorch
TORCH_LIBRARY(myop,  m) {

m.def("custom_linear(Tensor weight, Tensor input, Tensor(?) bias) -> Tensor", custom_linear);

m.def("custom_linear.out(Tensor weight, Tensor input, Tensor(?) bias, *, Tensor(a!) out) -> Tensor(a!)", WRAP_TO_ATEN(custom_linear_out, 3));
}
```

### Compile and link the custom kernel

Link it into ExecuTorch runtime: In our runner CMakeLists.txt we just need to add custom_linear.h/cpp into the binary target. We can build a dynamically loaded library (.so or .dylib) and link it as well.



Link it into PyTorch runtime: We need to package custom_linear.h, custom_linear.cpp and custom_linear_pytorch.cpp into a dynamically loaded library (.so or .dylib) and load it into our python environment. One way of doing this is:

```
import torch
torch.ops.load_library("libcustom_linear.so/dylib")
```


Once loaded we can perform the next step, of introducing the custom op into PyTorch environment.

### Source-to-source transformation to introduce the custom op

Easier way to introduce our customized linear is by rewriting the eager model. However, that may miss some occurrences of torch.nn.Linear in our example. A safer option is to walk through all the modules in the module hierarchy and perform the swapping.

For example, we can do the following to swap torch.nn.Linear with our custom linear op:

```
def  replace_linear_with_custom_linear(module):
    for  name,  child  in  module.named_children():
        if  isinstance(child,  nn.Linear):
            setattr(
                module,
                name,
                CustomLinear(child.in_features,  child.out_features, child.bias),
        )
    else:
        replace_linear_with_custom_linear(child)
```

The rest of the steps will be the same as the normal flow. Now you can run this module in eager as well as export it to ExecuTorch and run on the runner.

## How to build Mobile Apps
You can also execute an LLM using ExecuTorch on iOS and Android

**For iOS details see the [iOS Sample App](https://github.com/pytorch/executorch/tree/main/examples/demo-apps/apple_ios).**


**For Android see the [Android Instructions](https://pytorch.org/executorch/main/llm/llama-demo-android.html).**
