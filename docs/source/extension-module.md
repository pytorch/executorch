# Running an ExecuTorch Model Using the Module Extension in C++

**Author:** [Anthony Shoumikhin](https://github.com/shoumikhin)

In the [Running an ExecuTorch Model in C++ Tutorial](running-a-model-cpp-tutorial.md), we explored the lower-level ExecuTorch APIs for running an exported model. While these APIs offer zero overhead, great flexibility, and control, they can be verbose and complex for regular use. To simplify this and resemble PyTorch's eager mode in Python, we introduce the Module facade APIs over the regular ExecuTorch runtime APIs. The Module APIs provide the same flexibility but default to commonly used components like `DataLoader` and `MemoryAllocator`, hiding most intricate details.

## Example

Let's see how we can run the `SimpleConv` model generated from the [Exporting to ExecuTorch tutorial](./tutorials/export-to-executorch-tutorial) using the `Module` APIs:

```cpp
#include <executorch/extension/module/module.h>

using namespace ::torch::executor;

// Create a Module.
Module module("/path/to/model.pte");

// Wrap the input data with a Tensor.
float input[1 * 3 * 256 * 256];
Tensor::SizesType sizes[] = {1, 3, 256, 256};
TensorImpl tensor(ScalarType::Float, std::size(sizes), sizes, input);

// Perform an inference.
const auto result = module.forward({EValue(Tensor(&tensor))});

// Check for success or failure.
if (result.ok()) {
  // Retrieve the output data.
  const auto output = result->at(0).toTensor().const_data_ptr<float>();
}
```

The code now boils down to creating a `Module` and calling `forward()` on it, with no additional setup. Let's take a closer look at these and other `Module` APIs to better understand the internal workings.

## APIs

### Creating a Module

Creating a `Module` object is an extremely fast operation that does not involve significant processing time or memory allocation. The actual loading of a `Program` and a `Method` happens lazily on the first inference unless explicitly requested with a dedicated API.

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
Note: the `Program` is loaded automatically before any `Method` is loaded. Subsequent attemps to load them have no effect if one of the previous attemps was successful.

### Querying for Metadata

Get a set of method names that a Module contains udsing the `method_names()` function:

```cpp
const auto method_names = module.method_names();

if (method_names.ok()) {
  assert(method_names.count("forward"));
}
```

Note: `method_names()` will try to force-load the `Program` when called the first time.

Introspect miscellaneous metadata about a particular method via `MethodMeta` struct returned by `method_meta()` function:

```cpp
const auto method_meta = module.method_meta("forward");

if (method_meta.ok()) {
  assert(method_meta->name() == "forward");
  assert(method_meta->num_inputs() > 1);

  const auto input_meta = method_meta->input_tensor_meta(0);

  if (input_meta.ok()) {
    assert(input_meta->scalar_type() == ScalarType::Float);
  }
  const auto output_meta = meta->output_tensor_meta(0);

  if (output_meta.ok()) {
    assert(output_meta->sizes().size() == 1);
  }
}
```

Note: `method_meta()` will try to force-load the `Method` when called for the first time.

### Perform an Inference

Assuming that the `Program`'s method names and their input format is known ahead of time, we rarely need to query for those and can run the methods directly by name using the `execute()` function:

```cpp
const auto result = module.execute("forward", {EValue(Tensor(&tensor))});
```

Which can also be simplified for the standard `forward()` method name as:

```cpp
const auto result = module.forward({EValue(Tensor(&tensor))});
```

Note: `execute()` or `forward()` will try to force load the `Program` and the `Method` when called for the first time. Therefore, the first inference will take more time than subsequent ones as it loads the model lazily and prepares it for execution unless the `Program` or `Method` was loaded explicitly earlier using the corresponding functions.

### Result and Error Types

Most of the ExecuTorch APIs, including those described above, return either `Result` or `Error` types. Let's understand what those are:

* [`Error`](https://github.com/pytorch/executorch/blob/main/runtime/core/error.h) is a C++ enum containing a collection of valid error codes, where the default is `Error::Ok`, denoting success.

* [`Result`](https://github.com/pytorch/executorch/blob/main/runtime/core/result.h) can hold either an `Error` if the operation has failed or a payload, i.e., the actual result of the operation like an `EValue` wrapping a `Tensor` or any other standard C++ data type if the operation succeeded. To check if `Result` has a valid value, call the `ok()` function. To get the `Error` use the `error()` function, and to get the actual data, use the overloaded `get()` function or dereferencing pointer operators like `*` and `->`.

### Profile the Module

Use [ExecuTorch Dump](sdk-etdump.md) to trace model execution. Create an instance of the `ETDumpGen` class and pass it to the `Module` constructor. After executing a method, save the `ETDump` to a file for further analysis. You can capture multiple executions in a single trace if desired.

```cpp
#include <fstream>
#include <memory>
#include <executorch/extension/module/module.h>
#include <executorch/sdk/etdump/etdump_flatcc.h>

using namespace ::torch::executor;

Module module("/path/to/model.pte", Module::LoadMode::MmapUseMlock, std::make_unique<ETDumpGen>());

// Execute a method, e.g. module.forward(...); or module.execute("my_method", ...);

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
