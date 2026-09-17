# CUDA Backend

The CUDA backend is the ExecuTorch solution for running models on NVIDIA GPUs. It leverages the [AOTInductor](https://pytorch.org/docs/stable/torch.compiler_aot_inductor.html) compiler to generate optimized CUDA kernels with libtorch-free execution, and uses [Triton](https://triton-lang.org/) for high-performance GPU kernel generation.

## Features

- **Optimized GPU Execution**: Uses AOTInductor to generate highly optimized CUDA kernels for model operators
- **Triton Kernel Support**: Leverages Triton for GEMM (General Matrix Multiply), convolution, and SDPA (Scaled Dot-Product Attention) kernels.
- **Quantization Support**: INT4 weight quantization with tile-packed format for improved performance and reduced memory footprint
- **Cross-Platform**: Supports both Linux and Windows platforms
- **Multiple Model Support**: Works with various models including LLMs, vision-language models, and audio models

## Target Requirements

Below are the requirements for running a CUDA-delegated ExecuTorch model:

- **Hardware**: NVIDIA GPU with CUDA compute capability
- **CUDA Toolkit**: CUDA 11.x or later (CUDA 12.x recommended)
- **Operating System**: Linux or Windows
- **Drivers**: PyTorch-Compatible NVIDIA GPU drivers installed

## Development Requirements

To develop and export models using the CUDA backend:

- **Python**: Python 3.8+
- **PyTorch**: PyTorch with CUDA support
- **ExecuTorch**: Install ExecuTorch with CUDA backend support

## Using the CUDA Backend

### Exporting Models with Python API

The CUDA backend uses the `CudaBackend` and `CudaPartitioner` classes to export models. Here is a complete example:

```python
import torch
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program

# Configure edge compilation
edge_compile_config = EdgeCompileConfig(
    _check_ir_validity=False,
    _skip_dim_order=True,
)

# Define your model
model = YourModel().eval()
example_inputs = (torch.randn(1, 3, 224, 224),)

# Export the model using torch.export
exported_program = torch.export.export(model, example_inputs)

# Create the CUDA partitioner
partitioner = CudaPartitioner(
    [CudaBackend.generate_method_name_compile_spec(model_name)]
)

# Add decompositions for Triton to generate kernels
exported_program = exported_program.run_decompositions({
    torch.ops.aten.conv1d.default: conv1d_to_conv2d,
})

# Lower to ExecuTorch with CUDA backend
et_program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[partitioner],
    compile_config=edge_compile_config,
)

# Convert to executable program and save
exec_program = et_program.to_executorch()
save_pte_program(exec_program, model_name, "./output_dir")
```
This generates `.pte` and `.ptd` files that can be executed on CUDA devices.

For a complete working example, see the [CUDA export script](https://github.com/pytorch/executorch/blob/main/examples/cuda/scripts/export.py).


----

## Runtime Integration

To run the model on device, use the standard ExecuTorch runtime APIs. See [Running on Device](getting-started.md#running-on-device) for more information.

When building from source, pass `-DEXECUTORCH_BUILD_CUDA=ON` when configuring the CMake build to compile the CUDA backend.

```
# CMakeLists.txt
add_subdirectory("executorch")
...
target_link_libraries(
    my_target
    PRIVATE executorch
    extension_module_static
    extension_tensor
    aoti_cuda_backend)
```

No additional steps are necessary to use the backend beyond linking the target. CUDA-delegated `.pte` and `.ptd` files will automatically run on the registered backend.

----

## Activation memory: who owns the GPU copies

A CUDA model has to get its input data onto the GPU somehow. There are two ways, and you pick
one at export time.

**Default: the runtime copies for you.** Export inserts a host-to-device copy in front of each
delegate input and a device-to-host copy after each delegate output. You pass ordinary CPU
tensors and the runtime moves the data:

```python
exec_program = et_program.to_executorch()
```

```python
# CPU tensors. The runtime copies them to the GPU and copies results back.
outputs = method([torch.randn(4, 64)])
```

This is the simplest option and the right default. The cost is a copy in each direction on
every call.

**Opt out: you hand over GPU memory directly.** If your data is already on the device, those
copies are wasted. Turn them off and the input placeholders are treated as device memory
instead:

```python
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.propagate_device_config import PropagateDeviceConfig

exec_program = et_program.to_executorch(
    ExecutorchBackendConfig(
        propagate_device_config=PropagateDeviceConfig(
            skip_h2d_for_method_inputs=True,
            skip_d2h_for_method_outputs=True,
        ),
        enable_non_cpu_memory_planning=True,
        # Required alongside the skips. Memory planning reserves a buffer for graph inputs and
        # outputs by default, and the runtime fills a planned input by copying the caller's
        # memory into that buffer, which reintroduces the copy you just asked to skip. On device
        # memory that copy is a host memcpy into a device pointer, which is undefined.
        memory_planning_pass=MemoryPlanningPass(
            alloc_graph_input=False, alloc_graph_output=False
        ),
    )
)
```

```python
# Now the tensors must already be on the GPU. Passing CPU tensors is a caller error.
outputs = method([torch.randn(4, 64, device="cuda")])
```

Outputs also stay on the device, which is what you want when the next stage of your pipeline is
also on the GPU.

### Things worth knowing

**The two flags are independent.** Skip only the input copies if you produce data on the GPU but
want results back on the host, or only the output copies for the reverse.

**Both flags need `enable_non_cpu_memory_planning=True`.** It defaults to `True`, so leaving it
out works, and the snippet above sets it explicitly only to make the requirement visible. Setting
it to `False` while asking for either skip raises a `ValueError`, because copy insertion happens
during device-aware memory planning.

**Both flags also need unplanned graph inputs and outputs**, via
`MemoryPlanningPass(alloc_graph_input=False, alloc_graph_output=False)` as shown above. Without
it the program still reserves its own buffer and the runtime copies into it, so the copy comes
back at run time. On device memory that copy is a host memcpy into a device pointer, which is
undefined, so the program crashes rather than returning a wrong answer.

**Per-method selection is on the outer config, not on the flags.** Pass a dict of
`PropagateDeviceConfig` keyed by method name:

```python
ExecutorchBackendConfig(
    propagate_device_config={
        "forward": PropagateDeviceConfig(
            skip_h2d_for_method_inputs=True, skip_d2h_for_method_outputs=True
        ),
        "forward_from_host": PropagateDeviceConfig(),
    },
    ...
)
```

Do not pass a dict to `skip_h2d_for_method_inputs` itself. The pass reads that field for
truthiness, so any non-empty dict enables the skip for every method regardless of the values
inside it.

**The choice is baked into the `.pte`.** A program exported with copies expects host tensors; one
exported without them expects device tensors. Passing the wrong kind is a caller error, not
something the runtime corrects, so the two are not interchangeable at run time.

**From C++ it is the same contract.** With the copies skipped, the input has to be a tensor
the delegate will accept as device resident. There are two ways to build one, and which is
right depends on where the data already is.

If the data is already in device memory, wrap the pointer with `from_blob` and name the
device right after the scalar type. Nothing is copied:

```cpp
auto input = from_blob(device_data, {rows, columns}, ScalarType::Float, DeviceType::CUDA);
auto result = module.forward(input);
```

This is the path to use when the producer of the data is another GPU kernel, a camera or
video decoder that writes to the GPU, or an earlier model whose output you kept on device.
`from_blob` does not allocate or migrate anything, so the pointer has to be genuinely valid
on the device you name, and it has to outlive the tensor.

If the data starts on the host, `clone_tensor_ptr_to` allocates on the device and copies it
across, which costs one transfer:

```cpp
auto host_input = make_tensor_ptr({rows, columns}, std::move(data));
auto input = clone_tensor_ptr_to(host_input, DeviceType::CUDA);
auto result = module.forward(input);
```

The device argument defaults to `DeviceType::CPU` in the overloads that do not take a custom
deleter, so a plain `from_blob(host_ptr, sizes, type)` still means host memory. The deleter
overloads require the device to be named. Leaving the device at CPU while handing over a
device pointer produces a CPU-tagged tensor that the delegate rejects.

**How the delegate decides a tensor is device resident.** Two checks, in this order. First
the tensor's own `device_type` has to be `CUDA`, which is metadata set by whoever built the
tensor. Then, because that tag is only a claim, the delegate calls
`cudaPointerGetAttributes` on the data pointer and requires the memory to really be
`cudaMemoryTypeDevice` or `cudaMemoryTypeManaged`. A CUDA-tagged tensor backed by host
memory is caught there rather than corrupting the run.

With the default export, pass an ordinary host tensor instead and the runtime handles the
transfer.

----

## Examples

For complete end-to-end examples of exporting and running models with the CUDA backend, see:

- [Whisper](https://github.com/pytorch/executorch/blob/main/examples/models/whisper/README.md) — Audio transcription model with CUDA support
- [Voxtral](https://github.com/pytorch/executorch/blob/main/examples/models/voxtral/README.md) — Audio multimodal model with CUDA support
- [Gemma3](https://github.com/pytorch/executorch/blob/main/examples/models/gemma3/README.md) — Vision-language model with CUDA support

These examples demonstrate the full workflow including model export, quantization options, building runners, and runtime execution.

ExecuTorch provides Makefile targets for building these example runners:

```bash
make whisper-cuda   # Build Whisper runner with CUDA
make voxtral-cuda   # Build Voxtral runner with CUDA
make gemma3-cuda    # Build Gemma3 runner with CUDA
```
