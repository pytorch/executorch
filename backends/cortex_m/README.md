# Cortex-M Backend

> [!NOTE]
> Beta. This backend has been validated with a set of small models (e.g. MLPerf Tiny, MobileNetV2) and provides broad operator coverage through CMSIS-NN accelerated kernels with portable-ops fallback.

## Overview

The Cortex-M backend is implemented as an operator dialect/library based on [CMSIS-NN](https://github.com/ARM-software/CMSIS-NN), together with the `CortexMQuantizer` which targets supported ops, and the `CortexMPassManager` which modifies the exported program to use Cortex-M operators where possible.

The default AOT path retains the established channels-last input and dim-order contract. An experimental explicit-layout path accepts ordinary contiguous inputs, inserts graph-visible NHWC copies, and lowers spatial kernels to the `cortex_m::*_nhwc` operator family. Enable it with `--cortex-m-explicit-layout`; the two modes do not fall back to or mix with each other. Explicit-layout compilation fails when a spatial operator is not eligible for NHWC lowering, so models using unsupported configurations must use the legacy mode.

### Explicit-layout migration

The `use_explicit_layout=True` modes on `CortexMQuantizer` and `CortexMPassManager` are temporary staging APIs. They keep the experimental path isolated while the default modes and `CortexMTester` continue to exercise the legacy path.

When explicit layout becomes the default, its support table and pass list will become the defaults in `CortexMQuantizer` and `CortexMPassManager`. The existing legacy support table and pass list will remain temporarily behind an opt-out AOT flag. This keeps the public Python entry points stable and switches `CortexMTester` to explicit layout without changing its callers. The direct NHWC kernel tests and explicit-only model tests will then be folded into the normal operator and model suites.

After the legacy AOT compatibility period, the legacy support table, pass list, input conversion, opt-out flag, and remaining dual-mode tests will be removed. Legacy runtime operators will remain registered so programs serialized by the old AOT path continue to load.

For a detailed example of the full lowering flow, see `examples/arm/cortex_m_mv2_example.ipynb`.

## Testing
Tests are available in `backends/cortex_m/test/` using the `backends/test` harness. The python implementations of the operators are tested in tests named `test_dialect_*`, while actual accelerated implementations are tested on simulated hardware in the tests named `test_implementation_*`.

To run tests:
```
examples/arm/setup.sh --i-agree-to-the-contained-eula                     # Download needed toolchains and simulators
examples/arm/arm-scratch/setup_path.sh                                    # Add dependencies to path
backends/cortex_m/test/build_test_runner.sh                               # Build executor-runner with cortex-m oplib + kernels registred
pytest --config-file=backends/arm/test/pytest.ini backends/cortex_m/test  # Run tests with correct configuration file
```

For an end-to-end bundled-IO FVP run of a single model (export → build → FVP → `Test_result: PASS`), use `examples/arm/run.sh`:
```
examples/arm/run.sh --model_name=<model> --target=cortex-m55+int8 --bundleio
```
This drives `aot_arm_compiler --bundleio`, invokes `build_test_runner.sh`, and launches the Corstone-300 FVP via `backends/arm/scripts/run_fvp.sh`.

## Supported operators
Refer to `backends/cortex_m/test/ops` for currently supported accelerated ops/dtypes. Additionally, the quantizer targets pure "data-movement ops" such as data copies, slicing and concatenations to use quantized dtypes using the portable-kernels operator library.
In general however, operators not supported by Cortex-M are kept in `fp32` using non-accelerated portable-kernels. It is recommended to analyze the graph after lowering to understand how much of the graph has been accelerated.

## Notices
Arm and Cortex are registered trademarks of Arm Limited (or its subsidiaries) in the US and/or elsewhere.
