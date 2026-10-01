# ExecuTorch Python Module (WIP)
This Python module, named `portable_lib`, provides a set of functions and classes for loading and executing bundled programs. To install it, run the fullowing command:

```bash
./install_executorch.sh

# ...or use pip directly
pip install . --no-build-isolation
```

## Portable tensor mode

Portable bindings do not link against ATen or PyTorch. CPU inputs may be NumPy
arrays or other dense objects implementing Python's buffer protocol. If
PyTorch is installed, `torch.Tensor` inputs are inspected through their Python
attributes instead of being cast to `at::Tensor`.

For compatibility with the existing API, importing `portable_lib` imports
PyTorch and portable bindings create output tensors through its public Python
API. The extension itself does not link against ATen or PyTorch.

To run without PyTorch, select `ExecuTorchResult` outputs before importing the
bindings. The setting is fixed when the extension loads, so importing PyTorch
later does not change the output type.

```python
import os

os.environ["EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT"] = "executorch"
from executorch.extension.pybindings import portable_lib
```

`ExecuTorchResult` exposes shape, strides, dtype, and the Python buffer protocol
so NumPy can read it without another copy. Without the environment setting,
failure to import PyTorch raises an error explaining how to opt into these
outputs. ATen-mode bindings continue to return native PyTorch tensors directly.

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
All functions and methods are guarded by a call guard that redirects `cout` and `cerr` to the Python environment.
