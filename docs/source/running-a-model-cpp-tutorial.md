# Running an ExecuTorch Model in C++ Tutorial

**Author:** [Jacob Szwejbka](https://github.com/JacobSzwejbka)

In this tutorial, we will cover how to run an ExecuTorch model in C++ using the more detailed, lower-level APIs: prepare the `MemoryManager`, set inputs, execute the model, and retrieve outputs. However, if you’re looking for a simpler interface that works out of the box, consider trying the [Module Extension Tutorial](extension-module.md).

For a high level overview of the ExecuTorch Runtime please see [Runtime Overview](runtime-overview.md), and for more in-depth documentation on
each API please see the [Runtime API Reference](executorch-runtime-api-reference.rst).
[Here](https://github.com/pytorch/executorch/blob/main/examples/portable/executor_runner/executor_runner.cpp) is a fully functional version C++ model runner, and the [Setting up ExecuTorch](getting-started-setup.md) doc shows how to build and run it.


## Prerequisites

You will need an ExecuTorch model to follow along. We will be using
the model `SimpleConv` generated from the [Exporting to ExecuTorch tutorial](./tutorials/export-to-executorch-tutorial).

## Model Loading

The first step towards running your model is to load it. ExecuTorch uses an abstraction called a `DataLoader` to handle the specifics of retrieving the `.pte` file data, and then `Program` represents the loaded state.

Users can define their own `DataLoader`s to fit the needs of their particular system. In this tutorial we will be using the `FileDataLoader`, but you can look under [Example Data Loader Implementations](https://github.com/pytorch/executorch/tree/main/extension/data_loader) to see other options provided by the ExecuTorch project.

For the `FileDataLoader` all we need to do is provide a file path to the constructor.

``` cpp
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::FileDataLoader;
using executorch::extension::MallocMemoryAllocator;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;

Result<FileDataLoader> loader =
        FileDataLoader::from("/tmp/model.pte");
assert(loader.ok());

Result<Program> program = Program::load(&loader.get());
assert(program.ok());
```

## Setting Up the MemoryManager

Next we will set up the `MemoryManager`.

One of the principles of ExecuTorch is giving users control over where the memory used by the runtime comes from. Today (late 2023) users need to provide 2 different allocators:

* Method Allocator: A `MemoryAllocator` used to allocate runtime structures at `Method` load time. Things like Tensor metadata, the internal chain of instructions, and other runtime state come from this.

* Planned Memory: A `HierarchicalAllocator` containing 1 or more memory arenas where internal mutable tensor data buffers are placed. At `Method` load time internal tensors have their data pointers assigned to various offsets within. The positions of those offsets and the sizes of the arenas are determined by memory planning ahead of time.

For this example we will retrieve the size of the planned memory arenas dynamically from the `Program`, but for heapless environments users could retrieve this information from the `Program` ahead of time and allocate the arena statically. We will also be using a malloc based allocator for the method allocator.

``` cpp
// Method names map back to Python nn.Module method names. Most users will only
// have the singular method "forward".
const char* method_name = "forward";

// MethodMeta is a lightweight structure that lets us gather metadata
// information about a specific method. In this case we are looking to get the
// required size of the memory planned buffers for the method "forward".
Result<MethodMeta> method_meta = program->method_meta(method_name);
assert(method_meta.ok());

std::vector<std::unique_ptr<uint8_t[]>> planned_buffers; // Owns the Memory
std::vector<Span<uint8_t>> planned_arenas; // Passed to the allocator

size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers();

// It is possible to have multiple layers in our memory hierarchy; for example,
// SRAM and DRAM.
for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
  // .get() will always succeed because id < num_memory_planned_buffers.
  size_t buffer_size =
      static_cast<size_t>(method_meta->memory_planned_buffer_size(id).get());
  planned_buffers.push_back(std::make_unique<uint8_t[]>(buffer_size));
  planned_arenas.push_back({planned_buffers.back().get(), buffer_size});
}
HierarchicalAllocator planned_memory(
    {planned_arenas.data(), planned_arenas.size()});

// Version of MemoryAllocator that uses malloc to handle allocations rather then
// a fixed buffer.
MallocMemoryAllocator method_allocator;

// Assemble all of the allocators into the MemoryManager that the Executor will
// use.
MemoryManager memory_manager(&method_allocator, &planned_memory);
```

## Loading a Method

In ExecuTorch we load and initialize from the `Program` at a method granularity. Many programs will only have one method 'forward'. `load_method` is where initialization is done, from setting up tensor metadata, to intializing delegates, etc.

``` cpp
Result<Method> method = program->load_method(method_name);
assert(method.ok());
```

## Setting Inputs

Now that we have our method we need to set up its inputs before we can
perform an inference. In this case we know our model takes a single (1, 3, 256, 256)
sized float tensor.

Depending on how your model was memory planned, the planned memory may or may
not contain buffer space for your inputs and outputs.

If the outputs were not memory planned then users will need to set up the output data pointer with 'set_output_data_ptr'. In this case we will just assume our model was exported with inputs and outputs handled by the memory plan.

``` cpp
// Create our input tensor.
float data[1 * 3 * 256 * 256];
Tensor::SizesType sizes[] = {1, 3, 256, 256};
Tensor::DimOrderType dim_order = {0, 1, 2, 3};
TensorImpl impl(
    ScalarType::Float, // dtype
    4, // number of dimensions
    sizes,
    data,
    dim_order);
Tensor t(&impl);

// Implicitly casts t to EValue
Error set_input_error = method->set_input(t, 0);
assert(set_input_error == Error::Ok);
```

## Perform an Inference

Now that our method is loaded and our inputs are set we can perform an inference. We do this by calling `execute`.

``` cpp
Error execute_error = method->execute();
assert(execute_error == Error::Ok);
```

## Retrieve Outputs

Once our inference completes we can retrieve our output. We know that our model only returns a single output tensor. One potential pitfall here is that the output we get back is owned by the `Method`. Users should take care to clone their output before performing any mutations on it, or if they need it to have a lifespan separate from the `Method`.

``` cpp
EValue output = method->get_output(0);
assert(output.isTensor());
```

## Conclusion

This tutorial demonstrated how to run an ExecuTorch model using low-level runtime APIs, which offer granular control over memory management and execution. However, for most use cases, we recommend using the Module APIs, which provide a more streamlined experience without sacrificing flexibility. For more details, check out the [Module Extension Tutorial](extension-module.md).
