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
