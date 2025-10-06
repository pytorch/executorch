# ExecuTorch Arm&reg; Backend

This subtree contains the Arm&reg; Backend implementation for ExecuTorch.
It supports multiple targets using a common infrastructure, that lowers
PyTorch models to a TOSA representation. This representation is used to
deploy to the following targets:

- **Arm&reg; Ethos&trade;-U55/65/85** - Compiled using the Ethos-U Vela compiler.
- **VGF (Vulkan&reg; Graph Format)** – SPIR-V™ representation for Vulkan-capable devices.

The backend provides an ahead-of-time (AOT) flow, that produces a PTE file for your
chosen target. The AOT flow supports the following development operating systems:

- Linux aarch64
- Linux x86_64
- macOS&trade; with Apple&reg; Silicon

In addition, the following deployment paths are supported by this backend:

- Bare metal build of a reference runtime for Arm&reg; Cortex&reg;-M with Ethos-U acceleration:
  - Full testing is available in tree using Corstone&trade; Fixed Virtual Platforms (FVP).
- Linux target support for VGF capable targets, using the executor_runner.

More information on TOSA can be found here: https://www.mlplatform.org/tosa/tosa_spec.html.

## Directory Layout

Below is an overview of the key folder and files in this directory:

```
backends/arm/
│
├── _passes/                       # Graph transformation passes
│   ├── arm_pass_manager.py        # Defines ordering of graph transformations
│   └── *_pass.py                  # Graph transformation implementation
│
├── common/                        # Common functionality used across the backend
│
├── debug/                         # Debugging schema and functionality
│
├── ethosu/                        # Implementations of EthosUPartitioner and EthosUBackend
│
├── operator_support/              # Checks if operators can be partitioned
│
├── operators/                     # ATen → TOSA serialization
│   ├── node_visitor.py            # Defines base class for ATen → TOSA node visitors
│   └── op_*.py                    # Lowering implementations for individual operators
│
├── quantizer/                     # Quantization-related logic
│   ├── arm_quantizer.py           # EthosUQuantizer and VGFQuantizer definitions
│   └── quantization_annotator.py  # Defines how operators are annotated for quantization
│
├── runtime/                       # Backends for running inference on target devices
│   ├── ArmEthosUBackend.cpp
│   └── VGFBackend.cpp
│
├── scripts/                       # Auxiliary build, dependency installation and utility scripts
│
├── test/                          # Unit tests for the backend
│   ├── ops/                       # Operator level unit tests
│   ├── models/                    # Model level unit tests
│   └── tester/                    # Testing harnesses and utilities
│
├── third-party/                   # External dependencies
│
├── tosa/                          # Shared TOSA backend implementation and dialect
│
└── vgf/                           # Implementations of VgfPartitioner and VgfBackend
```

## Building

The Arm backend can be built using the following command:

```
./install_executorch.sh
```

One of the following commands should also be run once to gather the necessary dependencies for your chosen target(s):

For the Ethos-U target:

```
./examples/arm/setup.sh --i-agree-to-the-contained-eula
```

For the VGF target:

```
./examples/arm/setup.sh --disable-ethos-u-deps --enable-mlsdk-deps
```

For both Ethos-U & VGF targets:

```
./examples/arm/setup.sh --i-agree-to-the-contained-eula --enable-mlsdk-deps
```

**NOTE:** While developing, it can be convenient to use`./install_executorch.sh --editable`, which creates an editable installation of ExecuTorch.

## Testing

There are two approaches for running the tests for the Arm backend. This section will explain these two approaches:

### Using test_arm_baremetal.sh

The backend provides a script `backends/arm/test/test_arm_baremetal.sh`, which is used in the `trunk` CI workflow.
This approach is useful for checking your change against this workflow on your own machine.
These scripts also install the necessary dependencies to run the tests.
Below is an overview of some of the testing options this script provides:

| Command                                        | Description                                  |
| ---------------------------------------------- | -------------------------------------------- |
| `test_arm_baremetal.sh test_pytest`            | Runs all unit tests.                         |
| `test_arm_baremetal.sh test_pytest_ethosu_fvp` | Same as `test_pytest` but uses Corstone FVP. |
| `test_arm_baremetal.sh test_run_ethosu_fvp`    | Runs some models with Corstone FVP.          |
| `test_arm_baremetal.sh test_full_ethosu_fvp`   | Runs E2E model tests on Corstone FVP.        |
| `test_arm_baremetal.sh test_pytest_vkml`       | Runs all unit tests with Vulkan ML.          |
| `test_arm_baremetal.sh test_full_vkml`         | Run E2E models test with Vulkan ML.          |

For more information, please refer to the `backends/arm/test/test_arm_baremetal.sh` script.

### Using pytest

The Arm backend uses `pytest` to run the unit test suite in `backends/arm/test`.
This option offers flexibility, allowing a specific test or a particular subset of the testsuite to be run.
Below provides some examples of how to use it:

- To run all the unit tests run the following command:

  ```
  pytest -v -n auto backends/arm/test/
  ```

- To run a specific test in a file:

  ```
  pytest -v backends/arm/test/ops/test_add.py -k test_add_tensor_tosa_INT_3
  ```

#### Testing Dependencies

Some tests, with `u55`, `u85` and `vgf` in the name require external dependencies to run if you use `pytest`:

- When a test contains `u55` or `u85`, you must run the following to setup the executor_runner:
  ```
  ./backends/arm/scripts/build_executorch.sh
  ./backends/arm/test/setup_testing.sh
  ```
- When a test contains `vgf`, you must run the following to install the ML SDK:
  ```
  ./backends/arm/scripts/build_executorch.sh
  ./backends/arm/test/setup_testing_vkml.sh
  ```

In addition, some model tests in the Arm backend require third-party libraries or packages.
To run these tests, you need to install the required dependencies by running the script `examples/arm/setup.sh` with the flag `--setup-test-dependency`.

Please note that installing model test dependencies is a standalone process. When using the `--setup-test-dependency` flag,
the script will install only the necessary dependencies for model tests, skipping all other setup procedures.

## Using pre-commit

A pre-commit script is available in the backend to help developers. Follow the steps below to enable it:

```
cp backends/arm/scripts/pre-commit .git/hooks/
```

## Notes on model specific and optional passes

The current TOSA version does not support int64. However, int64 is commonly used in many models. In order to lower the operators with int64 inputs and/or outputs to TOSA, a few passes have been developed to handle the int64-related issues. The main idea behind these passes is to replace the uses of int64 with int32 where feasible.

- For floating-point models, these passes need to run very early in the lowering process and can be passed in to the to_edge_transform_and_lower() function call as an optional parameter.
- For quantized models, these transformations will be automatically handled during annotation before the export stage.

List of model specific and optional passes:

- ConvertInt64ConstOpsToInt32Pass
  - Functionalities:
    - Rewrites constant-producing ops that output int64 to instead output int32, when values are within int32 bounds.
  - Supported Ops:
    - `torch.full`, `torch.arange`, `torch.eye`, `torch.linspace`, `torch.tensor`
  - Example usage:
    - backends/arm/test/models/stable_diffusion/test_CLIPTextModelWithProjection.py
    - backends/arm/test/models/stable_diffusion/test_T5EncoderModel.py

- ConvertInt64OutputOpsToInt32Pass
  - Overview:
    - Rewrites or removes operations that produce int64 outputs, converting them to int32 where possible.
    - Overflow checks are applied selectively; for ops without such checks, users need to ensure values fit within the int32 range.
  - Functionalities:
    1. Handling casting to int64:
       - (1) int32 -> int64:
         - Removes the cast and redirect uses of int64 to int32
       - (2) other types -> int64:
         - Rewrites the cast to other types -> int32
       - Supported Ops:
         - torch.ops.aten.to.\[dtype|dtype_layout\]
         - exir_ops.edge.dim_order_ops.\_to_dim_order_copy.default
    2. Post-process argmax outputs:
       - Inserts an int64->int32 cast after the argmax operations that produce int64 outputs:
       - Supported Ops:
         - torch.ops.aten.argmax.default
         - exir_ops.edge.aten.argmax.default
  - Example usage:
    - (Functionality 1) backends/arm/test/models/stable_diffusion/test_T5EncoderModel.py
    - (Functionality 2) backends/arm/test/models/stable_diffusion/test_CLIPTextModelWithProjection.py

- InsertInt32CastsAfterInt64PlaceholdersPass
  - Functionalities:
    - Inserts an int64 -> int32 cast immediately after each int64 placeholder (graph input).
    - Redirects all uses of each int64 placeholder to its int32 cast output.
    - Inserts local int32 -> int64 casts at call sites where an operator requires int64 inputs, e.g. `torch.nn.functional.one_hot`
  - Pass ordering:
    - When used with `ConvertInt64ConstOpsToInt32Pass` and `ConvertInt64OutputOpsToInt32Pass`, run this pass last.
    - Rationale: Those passes may cause retracing to re-infer some int64 placeholders as int32. Running this pass last casts only inputs that remain int64, minimizing inserted casts.
  - Example usage:
    - backends/arm/test/models/test_llama.py
    - backends/arm/test/models/stable_diffusion/test_CLIPTextModelWithProjection.py
    - backends/arm/test/models/stable_diffusion/test_T5EncoderModel.py

## Help & Improvements

If you have problems or questions, or have suggestions for ways to improve the Arm backend, please reach out
to the Arm team developing this backend, or create an issue on [here](https://www.github.com/pytorch/executorch/issues) and add the "partner: arm" label.
