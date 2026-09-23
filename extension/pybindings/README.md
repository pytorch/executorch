# ExecuTorch Python Module (WIP)
This Python module, named `portable_lib`, provides a set of functions and classes for loading and executing bundled programs. To install it, run the fullowing command:

```bash
./install_executorch.sh

# ...or use pip directly
pip install . --no-build-isolation
```

## Lightweight Tensor API

In addition to `torch.Tensor`, the bindings accept `portable_lib.Tensor`, a
lightweight CPU tensor that can be created from a NumPy array or nested Python
list. Objects implementing Python's buffer protocol, such as NumPy arrays and
`memoryview`, may also be passed directly as tensor inputs and are copied into
lightweight tensor storage:

```python
import numpy as np
from executorch.extension.pybindings import portable_lib

x = portable_lib.Tensor(np.ones((2, 2), dtype=np.float32))
y = portable_lib.Tensor([[1, 2], [3, 4]], dtype=np.float32)
output = module((x, y))[0]
result = output.numpy()
```

`Tensor` also implements the read-only Python buffer protocol, so consumers can
read it with `memoryview(output)` or `numpy.asarray(output)`.

ATen-enabled builds accept both `torch.Tensor` and `portable_lib.Tensor` through
the same methods. Tensor outputs use the lightweight type when a lightweight
input is provided, including for mixed input lists; existing calls containing
only `torch.Tensor` inputs continue to return `torch.Tensor`. Lightweight tensor
outputs are always copied, so `clone_outputs` only affects `torch.Tensor`
outputs.

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
- `run_method()`: Run method.
- `forward()`: Forward. This takes a pytree-flattend PyTorch-tensor-based input.
- `has_etdump()`: Check if etdump is available.
- `write_etdump_result_to_file()`: Write etdump result to a file.
- `__call__()`: Call method.
### BundledModule
This class is currently empty and serves as a placeholder for future methods and attributes.
- `verify_result_with_bundled_expected_output(method_name: str, testset_idx: int, rtol: float = 1e-5, atol: float = 1e-8)`: Verify result with bundled expected output.
## Note
All functions and methods are guarded by a call guard that redirects `cout` and `cerr` to the Python environment.
