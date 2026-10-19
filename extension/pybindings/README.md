# ExecuTorch Python Module (WIP)
This Python module, named `portable_lib`, provides a set of functions and classes for loading and executing bundled programs. To install it, run the fullowing command:

```bash
./install_executorch.sh

# ...or use pip directly
pip install . --no-build-isolation
```

## Portable tensor mode

Portable bindings do not link against ATen or PyTorch. CPU inputs may be NumPy
arrays or other dense objects implementing Python's buffer protocol, including
`ctypes` arrays, `array.array`, and complex NumPy arrays. CPU and
CUDA `torch.Tensor` inputs are inspected through their public Python attributes
instead of being cast to `at::Tensor`. DLPack producers, including CuPy, can
provide CPU or CUDA allocations directly; their device indices are preserved.
Tensor protocols can be mixed within the same call.

Tensor inputs must have dense storage. In the portable bindings, an input laid
out in a different dimension order than the exported method, for example a
channels-last tensor for a contiguous method or a transposed array, is copied
into the exported layout: on the host for buffers and CPU DLPack tensors, and by
torch on the tensor's own device for torch tensors. Read-only buffers are
copied too, in any layout. A non-torch CUDA DLPack input must already be in the
exported layout. The ATen bindings do not convert inputs, so pass them in the
exported layout there.

Pass an input again after changing it. A method exported with the default
memory planning copies every input into its own memory, and a converted input
is a copy too. Only a method exported with
`MemoryPlanningPass(alloc_graph_input=False)` reads an unconverted, writable
input in place. CUDA inputs require a method
exported with CUDA inputs and `MemoryPlanningPass(alloc_graph_input=False)` so
execution can share the caller's device allocation.

When PyTorch is available, importing `portable_lib` imports it and portable
bindings return `torch.Tensor` outputs. When it cannot be imported, the default
output type is `ExecuTorchResult`. Output conversion uses Python APIs and
DLPack; the extension does not link against PyTorch.
`clone_outputs=True` returns independent storage on the output's original
device. With `clone_outputs=False`, results share runtime or input storage and
later execution may overwrite that storage.

A CUDA output that fills memory the bindings allocated for it is returned
without a copy when `clone_outputs=True`. The result keeps that memory, and the
next execution writes into new memory while the result is alive. Reading the
outputs again before then returns the same object.

`dtype_name` names a result's dtype, including dtypes NumPy lacks, such as
`"bfloat16"`.

Calls on a shared method or module are serialized. A sequence of separate
`set_inputs`, `execute`, and `get_outputs` calls requires caller synchronization
when each thread needs its own inputs and outputs.

To select `ExecuTorchResult` outputs even when PyTorch is installed, set the
environment variable before importing the bindings. The setting is fixed when
the extension loads, so importing PyTorch later does not change the output type.

```python
import os

os.environ["EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT"] = "executorch"
from executorch.extension.pybindings import portable_lib
```

`ExecuTorchResult` preserves shape, strides, dtype, and device. CPU results expose
the Python buffer protocol so NumPy can read them without another copy. Without
the environment setting, the bindings fall back to `ExecuTorchResult` if
PyTorch cannot be imported. Explicitly selecting `torch` requires it to import
successfully. ATen-mode bindings continue to return native PyTorch tensors
directly.

```python
output = module.forward([numpy_input])[0]
result = np.asarray(output)
```

CPU and CUDA results implement `__dlpack__` and `__dlpack_device__`. CUDA results
remain on CUDA and can be passed to another DLPack consumer or back into
ExecuTorch without moving the data to the CPU.

Without torch or another DLPack library, `to_host()` copies a result to CPU
memory, and `ExecuTorchResult.from_host()` copies a CPU buffer to a CUDA device
for a method whose inputs live there:

```python
from executorch.extension.pybindings.portable_lib import ExecuTorchResult

device_input = ExecuTorchResult.from_host(numpy_input, device_index=0)
output = method([device_input])[0]
result = np.asarray(output.to_host())
```

CUDA DLPack inputs and `ExecuTorchResult` outputs require CUDA-enabled bindings.
Torch CUDA inputs and outputs synchronize through Python Torch, including in
CPU builds of the bindings. Execution is synchronous at the Python boundary:
a call returns after its device work has finished, so its outputs are ready on
any stream. Each call waits for the device once, rather than once per tensor.
Before running, it also waits once if an input may still be written on another
library's stream: a torch CUDA tensor, a DLPack input from another library, or
an `ExecuTorchResult` whose memory was handed out through `__dlpack__`. The
synchronization has no C++ ATen dependency.

For bfloat16 inputs, raw byte buffers use the shape declared by the exported
method. Bfloat16 results expose flat bytes through the buffer protocol because
NumPy and PEP 3118 do not define that numeric dtype. DLPack consumers receive the
original bfloat16 dtype and tensor shape.

```python
import cupy as cp

cuda_input = cp.ones((1, 3, 224, 224), dtype=cp.float32)
output = cuda_module.forward([cuda_input], clone_outputs=False)[0]
cuda_output = cp.from_dlpack(output)
```

The CUDA build workflow runs `test_pybindings_cuda_dlpack.py` in both output
modes on a GPU. It checks device allocations, DLPack stream ordering, channels
last views, cloning, and borrowed output lifetimes using a CUDA delegate.
Set `EXECUTORCH_PYBIND_TEST_REQUIRE_CUDA=1` when running this test to treat missing
GPU hardware or a missing CUDA backend as a failure. The CUDA delegate's kernels
need the OpenMP runtime (`libgomp.so.1`). PyTorch ships one; without PyTorch,
install it from the system, or the two tests that run without PyTorch fail
under this setting.

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
- `run_method()`: Run a method with PyTorch tensors, Python buffer objects such
  as NumPy arrays, or DLPack producers, including mixed tensor protocols.
- `forward()`: Forward. A single buffer may be passed directly; multiple
  inputs are passed as a flat sequence.
- `has_etdump()`: Check if etdump is available.
- `write_etdump_result_to_file()`: Write etdump result to a file.
- `__call__()`: Call method.
### BundledModule
This class is currently empty and serves as a placeholder for future methods and attributes.
- `verify_result_with_bundled_expected_output(method_name: str, testset_idx: int, rtol: float = 1e-5, atol: float = 1e-8)`: Verify result with bundled expected output.
## Note
Runtime logs are forwarded to Python stderr.
