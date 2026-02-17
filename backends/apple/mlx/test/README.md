# MLX Backend Tests

This directory contains end-to-end tests for the MLX backend. Each test verifies that a specific op or pattern is correctly lowered to MLX and produces matching outputs between PyTorch and the MLX runtime.

## Setup

### 1. Install ExecuTorch Python package (if not already installed)

```bash
python install_executorch.py --editable
```

### 2. Configure CMake with MLX preset

From the ExecuTorch root directory:

```bash
cmake --preset mlx-release
```

This configures the build with MLX delegate support. Build files are generated in `cmake-out/`.

### 3. Build the test runner

```bash
cmake --build cmake-out --target op_test_runner
```

This builds the `op_test_runner` binary that executes `.pte` models using the MLX runtime.



## Prerequisites

1. **Python environment**: Tests must be run in an environment where the `executorch` Python package is installed
2. **Built C++ runtime**: The `op_test_runner` binary must be built (see Setup above)

## Running Tests

### Run All Tests

To run all registered tests:

```bash
python -m executorch.backends.apple.mlx.test.run_all_tests -j4 --clean-after
```

Options:
- `-j N` / `--parallel N`: Run tests in parallel with N workers
- `--clean-after`: Clean up generated test files after running
- `--rebuild`: Rebuild the C++ test runner before running (use after C++ runtime changes)
- `--list`: List available tests and exit
- `-v` / `--verbose`: Verbose output

### Run a Specific Test

To run a specific test by name (e.g., `linear`):

```bash
python -m executorch.backends.apple.mlx.test.run_all_tests linear
```

With verbose output:

```bash
python -m executorch.backends.apple.mlx.test.run_all_tests -v linear
```

### List Available Tests

```bash
python -m executorch.backends.apple.mlx.test.run_all_tests --list
```

## Test Architecture

All tests are defined in `test_ops.py`. Each test follows a common pattern:

1. **Define a model** - A simple `nn.Module` that uses the op being tested
2. **Create test inputs** - Generate random input tensors
3. **Export and lower** - Export the model and lower it to the MLX backend
4. **Run C++ binary** - Execute the lowered model using `op_test_runner`
5. **Compare outputs** - Verify PyTorch and MLX outputs match within tolerance

### Test Class Structure

Tests inherit from `OpTestCase` and implement:

```python
@register_test
class MyTest(OpTestCase):
    name = "my_test"           # Test name (used for output directory)
    rtol = 1e-5                # Relative tolerance for comparison
    atol = 1e-5                # Absolute tolerance for comparison

    def create_model(self) -> nn.Module:
        """Return the model to test."""
        ...

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Return input tensors for export."""
        ...

    def get_dynamic_shapes(self) -> Optional[Dict]:
        """Return dynamic shape specs, or None for static shapes."""
        ...

    @classmethod
    def get_test_configs(cls) -> List["MyTest"]:
        """Return list of test configurations to run."""
        ...
```

## Test Output

Test artifacts are saved to `op_tests/<test_name>/`:
- `model.pte` - Exported ExecuTorch model
- `input.bin` - Serialized input tensors
- `expected_output.bin` - PyTorch reference output
- `actual_output.bin` - MLX runtime output

## Adding a New Test

1. Add a new model class and `OpTestCase` subclass to `test_ops.py`
2. Use the `@register_test` decorator on the test class
3. Implement `create_model()`, `create_inputs()`, and `get_test_configs()`
4. Run the test to verify it works E2E
