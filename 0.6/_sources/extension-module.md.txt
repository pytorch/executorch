# Running an ExecuTorch Model Using the Module Extension in C++

**Author:** [Anthony Shoumikhin](https://github.com/shoumikhin)

In the [Running an ExecuTorch Model in C++ Tutorial](running-a-model-cpp-tutorial.md), we explored the lower-level ExecuTorch APIs for running an exported model. While these APIs offer zero overhead, great flexibility, and control, they can be verbose and complex for regular use. To simplify this and resemble PyTorch's eager mode in Python, we introduce the `Module` facade APIs over the regular ExecuTorch runtime APIs. The `Module` APIs provide the same flexibility but default to commonly used components like `DataLoader` and `MemoryAllocator`, hiding most intricate details.

## Example

Let's see how we can run the `SimpleConv` model generated from the [Exporting to ExecuTorch tutorial](./tutorials/export-to-executorch-tutorial) using the `Module` and [`TensorPtr`](extension-tensor.md) APIs:

```cpp
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

using namespace ::executorch::extension;

// Create a Module.
Module module("/path/to/model.pte");

// Wrap the input data with a Tensor.
float input[1 * 3 * 256 * 256];
auto tensor = from_blob(input, {1, 3, 256, 256});

// Perform an inference.
const auto result = module.forward(tensor);

// Check for success or failure.
if (result.ok()) {
  // Retrieve the output data.
  const auto output = result->at(0).toTensor().const_data_ptr<float>();
}
```

The code now boils down to creating a `Module` and calling `forward()` on it, with no additional setup. Let's take a closer look at these and other `Module` APIs to better understand the internal workings.

## APIs

### Creating a Module

Creating a `Module` object is a fast operation that does not involve significant processing time or memory allocation. The actual loading of a `Program` and a `Method` happens lazily on the first inference unless explicitly requested with a dedicated API.

```cpp
Module module("/path/to/model.pte");
```

### Force-Loading a Method

To force-load the `Module` (and thus the underlying ExecuTorch `Program`) at any time, use the `load()` function:

```cpp
const auto error = module.load();

assert(module.is_loaded());
```

To force-load a particular `Method`, call the `load_method()` function:

```cpp
const auto error = module.load_method("forward");

assert(module.is_method_loaded("forward"));
```

You can also use the convenience function to load the `forward` method:

```cpp
const auto error = module.load_forward();

assert(module.is_method_loaded("forward"));
```

**Note:** The `Program` is loaded automatically before any `Method` is loaded. Subsequent attempts to load them have no effect if a previous attempt was successful.

### Querying for Metadata

Get a set of method names that a `Module` contains using the `method_names()` function:

```cpp
const auto method_names = module.method_names();

if (method_names.ok()) {
  assert(method_names->count("forward"));
}
```

**Note:** `method_names()` will force-load the `Program` when called for the first time.

To introspect miscellaneous metadata about a particular method, use the `method_meta()` function, which returns a `MethodMeta` struct:

```cpp
const auto method_meta = module.method_meta("forward");

if (method_meta.ok()) {
  assert(method_meta->name() == "forward");
  assert(method_meta->num_inputs() > 1);

  const auto input_meta = method_meta->input_tensor_meta(0);
  if (input_meta.ok()) {
    assert(input_meta->scalar_type() == ScalarType::Float);
  }

  const auto output_meta = method_meta->output_tensor_meta(0);
  if (output_meta.ok()) {
    assert(output_meta->sizes().size() == 1);
  }
}
```

**Note:** `method_meta()` will also force-load the `Method` the first time it is called.

### Performing an Inference

Assuming the `Program`'s method names and their input format are known ahead of time, you can run methods directly by name using the `execute()` function:

```cpp
const auto result = module.execute("forward", tensor);
```

For the standard `forward()` method, the above can be simplified:

```cpp
const auto result = module.forward(tensor);
```

**Note:** `execute()` or `forward()` will load the `Program` and the `Method` the first time they are called. Therefore, the first inference will take longer, as the model is loaded lazily and prepared for execution unless it was explicitly loaded earlier.

### Setting Input and Output

You can set individual input and output values for methods with the following APIs.

#### Setting Inputs

Inputs can be any `EValue`, which includes tensors, scalars, lists, and other supported types. To set a specific input value for a method:

```cpp
module.set_input("forward", input_value, input_index);
```

- `input_value` is an `EValue` representing the input you want to set.
- `input_index` is the zero-based index of the input to set.

For example, to set the first input tensor:

```cpp
module.set_input("forward", tensor_value, 0);
```

You can also set multiple inputs at once:

```cpp
std::vector<runtime::EValue> inputs = {input1, input2, input3};
module.set_inputs("forward", inputs);
```

**Note:** You can skip the method name argument for the `forward()` method.

By pre-setting all inputs, you can perform an inference without passing any arguments:

```cpp
const auto result = module.forward();
```

Or just setting and then passing the inputs partially:

```cpp
// Set the second input ahead of time.
module.set_input(input_value_1, 1);

// Execute the method, providing the first input at call time.
const auto result = module.forward(input_value_0);
```

**Note:** The pre-set inputs are stored in the `Module` and can be reused multiple times for the next executions.

Don't forget to clear or reset the inputs if you don't need them anymore by setting them to default-constructed `EValue`:

```cpp
module.set_input(runtime::EValue(), 1);
```

#### Setting Outputs

Only outputs of type Tensor can be set at runtime, and they must not be memory-planned at model export time. Memory-planned tensors are preallocated during model export and cannot be replaced.

To set the output tensor for a specific method:

```cpp
module.set_output("forward", output_tensor, output_index);
```

- `output_tensor` is an `EValue` containing the tensor you want to set as the output.
- `output_index` is the zero-based index of the output to set.

**Note:** Ensure that the output tensor you're setting matches the expected shape and data type of the method's output.

You can skip the method name for `forward()` and the index for the first output:

```cpp
module.set_output(output_tensor);
```

**Note:** The pre-set outputs are stored in the `Module` and can be reused multiple times for the next executions, just like inputs.

### Result and Error Types

Most of the ExecuTorch APIs return either `Result` or `Error` types:

- [`Error`](https://github.com/pytorch/executorch/blob/main/runtime/core/error.h) is a C++ enum containing valid error codes. The default is `Error::Ok`, denoting success.

- [`Result`](https://github.com/pytorch/executorch/blob/main/runtime/core/result.h) can hold either an `Error` if the operation fails, or a payload such as an `EValue` wrapping a `Tensor` if successful. To check if a `Result` is valid, call `ok()`. To retrieve the `Error`, use `error()`, and to get the data, use `get()` or dereference operators like `*` and `->`.

### Profiling the Module

Use [ExecuTorch Dump](etdump.md) to trace model execution. Create an `ETDumpGen` instance and pass it to the `Module` constructor. After executing a method, save the `ETDump` data to a file for further analysis:

```cpp
#include <fstream>
#include <memory>

#include <executorch/extension/module/module.h>
#include <executorch/devtools/etdump/etdump_flatcc.h>

using namespace ::executorch::extension;

Module module("/path/to/model.pte", Module::LoadMode::MmapUseMlock, std::make_unique<ETDumpGen>());

// Execute a method, e.g., module.forward(...); or module.execute("my_method", ...);

if (auto* etdump = dynamic_cast<ETDumpGen*>(module.event_tracer())) {
  const auto trace = etdump->get_etdump_data();

  if (trace.buf && trace.size > 0) {
    std::unique_ptr<void, decltype(&free)> guard(trace.buf, free);
    std::ofstream file("/path/to/trace.etdump", std::ios::binary);

    if (file) {
      file.write(static_cast<const char*>(trace.buf), trace.size);
    }
  }
}
```

## Conclusion

The `Module` APIs provide a simplified interface for running ExecuTorch models in C++, closely resembling the experience of PyTorch's eager mode. By abstracting away the complexities of the lower-level runtime APIs, developers can focus on model execution without worrying about the underlying details.
