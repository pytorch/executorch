# MLX Backend Tests

This directory contains end-to-end tests for the MLX backend. Each test verifies that a specific op or pattern is correctly lowered to MLX and produces matching outputs between PyTorch and the MLX runtime.

## Prerequisites

1. **Python environment**: Tests must be run in an environment where the `executorch` Python package is installed
2. **Built C++ runtime**: The `op_test_runner` binary must be built (see main MLX backend README)

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

### Run a Specific Test

To run a specific test (e.g., `linear`):

```bash
python -m executorch.backends.apple.mlx.test.test_linear run
```

With verbose output:

```bash
python -m executorch.backends.apple.mlx.test.test_linear run --verbose
```

### Run Test Variants

Some tests have multiple configurations. For example, `test_permute` tests both `permute` and `transpose`:

```bash
# Run permute variant (default)
python -m executorch.backends.apple.mlx.test.test_permute run

# Run transpose variant
python -m executorch.backends.apple.mlx.test.test_permute run --variant transpose
```

### Custom Test Parameters

Most tests accept custom parameters:

```bash
# Linear with custom dimensions
python -m executorch.backends.apple.mlx.test.test_linear run --in-features 128 --out-features 256

# Quantized linear with different group size
python -m executorch.backends.apple.mlx.test.test_quantized_linear run --group-size 64
```

Run with `--help` to see available options:

```bash
python -m executorch.backends.apple.mlx.test.test_linear run --help
```

## Available Tests

Test files are named `test_<op_name>.py`. To see all available tests, list the directory or run:

```bash
python -m executorch.backends.apple.mlx.test.run_all_tests --list
```

## Test Architecture

Each test follows a common pattern:

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

1. Create `test_<op_name>.py` following the pattern of existing tests
2. Implement the `OpTestCase` subclass with `@register_test` decorator
3. Add factory function `_create_from_args` and `_add_args` for CLI support
4. Run the test to verify it works E2E
