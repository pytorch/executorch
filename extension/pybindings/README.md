# ExecuTorch Python Module (WIP)
This Python module, named `portable_lib`, provides a set of functions and classes for loading and executing bundled programs. To install it, run the fullowing command:

```bash
./install_executorch.sh

# ...or use pip directly
pip install . --no-build-isolation
```

## Portable tensor mode

Portable bindings do not link against ATen or PyTorch. CPU inputs may be NumPy
arrays or other dense objects implementing Python's buffer protocol. CPU and
CUDA `torch.Tensor` inputs are inspected through their public Python attributes
instead of being cast to `at::Tensor`; CUDA pointers retain their device index.

Portable Torch inputs must have dense strides compatible with the exported
method input's dimension order. This includes channels last and channels last
3D layouts when the model was exported with those layouts. Strides on singleton
dimensions are normalized to the exported layout without copying tensor data.

When PyTorch is available, importing `portable_lib` imports it and portable
bindings create output tensors through its public Python API. When it cannot be
imported, the default output type is `ExecuTorchResult`. The extension itself does not link against ATen or PyTorch.
CUDA outputs use a private DLPack capsule consumed by `torch.from_dlpack`, and
`clone_outputs=True` copies them through Torch on their original CUDA device.
The `executorch` output mode supports CPU buffers; CUDA outputs require the
default Torch output mode.

To select `ExecuTorchResult` outputs even when PyTorch is installed, set the
environment variable before importing the bindings. The setting is fixed when the extension loads, so importing PyTorch
later does not change the output type.

```python
import os

os.environ["EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT"] = "executorch"
from executorch.extension.pybindings import portable_lib
```

`ExecuTorchResult` exposes shape, strides, dtype, and the Python buffer protocol
so NumPy can read it without another copy. Explicitly setting the environment
variable to `torch` requires PyTorch to import successfully. ATen-mode bindings continue to return native PyTorch tensors directly.

```python
output = module.forward([numpy_input])[0]
result = np.asarray(output)
```

# Link Backends

Not all backends are built into the pip wheel by default. You can link these missing/experimental backends by turning on the corresponding cmake flag. For example, to include the Vulkan backend:

```bash
CMAKE_ARGS="-DEXECUTORCH_BUILD_VULKAN=ON" ./install_executorch.sh
```

## Functions
- `_load_for_executorch(path: str, enable_etdump: bool = False)`: Load a module from a file.
- `_load_for_executorch_from_buffer(buffer: str, enable_etdump: bool = False)`: Load a module from a buffer.
- `_load_for_executorch_from_bundled_program(ptr: str, enable_etdump: bool = False)`: Load a module from a bundled program.
- `_load_bundled_program_from_buffer(buffer: str, non_const_pool_size: int = kDEFAULT_BUNDLED_INPUT_POOL_SIZE)`: Load a bundled program from a buffer.
- `_dump_profile_results()`: Dump profile results.
- `_get_operator_names()`: Get operator names.
- `_create_profile_block()`: Create a profile block.
- `_reset_profile_results()`: Reset profile results.
## Classes
### ExecuTorchModule
- `plan_execute()`: Plan and execute.
- `run_method()`: Run a method with either PyTorch tensors or objects that
  implement Python's buffer protocol, such as NumPy arrays. A call must use a
  single tensor protocol; PyTorch tensors and buffers cannot be mixed.
- `forward()`: Forward. A single buffer may be passed directly; multiple
  inputs are passed as a flat sequence.
- `has_etdump()`: Check if etdump is available.
- `write_etdump_result_to_file()`: Write etdump result to a file.
- `__call__()`: Call method.
### BundledModule
This class is currently empty and serves as a placeholder for future methods and attributes.
- `verify_result_with_bundled_expected_output(method_name: str, testset_idx: int, rtol: float = 1e-5, atol: float = 1e-8)`: Verify result with bundled expected output.
## Note
Calls on a shared method or module are serialized. A sequence of separate
`set_inputs`, `execute`, and `get_outputs` calls requires caller synchronization
when each thread needs its own inputs and outputs.
