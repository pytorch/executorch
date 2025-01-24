# ExecuTorch Python Module (WIP)
This Python module, named `portable_lib`, provides a set of functions and classes for loading and executing bundled programs. To install it, run the fullowing command:

```bash
EXECUTORCH_BUILD_PYBIND=ON \
pip install . --no-build-isolation
```

Or when installing the rest of dependencies:

```bash
install_requirements.sh --pybind
```

# Link Backends

You can link the runtime against some backends to make sure a delegated or partitioned model can still run by Python module successfully:

```bash
EXECUTORCH_BUILD_PYBIND=ON \
CMAKE_ARGS="-DEXECUTORCH_BUILD_COREML=ON -DEXECUTORCH_BUILD_MPS=ON -DEXECUTORCH_BUILD_XNNPACK=ON" \
pip install . --no-build-isolation
```

Similarly, when installing the rest of dependencies:

```bash
install_requirements.sh --pybind coreml mps xnnpack
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
- `load_bundled_input()`: Load bundled input.
- `verify_result_with_bundled_expected_output(bundle: str, method_name: str, testset_idx: int, rtol: float = 1e-5, atol: float = 1e-8)`: Verify result with bundled expected output.
- `plan_execute()`: Plan and execute.
- `run_method()`: Run method.
- `forward()`: Forward. This takes a pytree-flattend PyTorch-tensor-based input.
- `has_etdump()`: Check if etdump is available.
- `write_etdump_result_to_file()`: Write etdump result to a file.
- `__call__()`: Call method.
### BundledModule
This class is currently empty and serves as a placeholder for future methods and attributes.
## Note
All functions and methods are guarded by a call guard that redirects `cout` and `cerr` to the Python environment.
