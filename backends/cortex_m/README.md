# Cortex-M Backend

> [!NOTE]
> WIP. This is a temporary/placeholder backend for Cortex-M CPUs. It is not intended to be used in production, but rather as a proof of concept. Things will change without notice.

## Overview

The Cortex-M backend is implemented as an operator dialect/library based on [CMSIS-NN](https://github.com/ARM-software/CMSIS-NN), together with the `CortexMQuantizer` which targets supported ops, and the `CortexMPassManager` which modifies the exported program to use Cortex-M operators where possible. It is intended for use with **channels-last input**  since this is what the accelerated kernels are using.

For a detailed example of the full lowering flow, see `examples/arm/cortex_m_minimal_example.ipynb`.

## Testing
Tests are available in `backends/cortex-m/test/` using the `backends/test` harness. The python implementations of the operators are tested in tests named `test_dialect_*`, while actual accelerated implementations are tested on simulated hardware in the tests named `test_implementation_*`.

To run tests:
```
examples/arm/setup.sh --i-agree-to-the-contained-eula                     # Download needed toolchains and simulators
examples/arm/arm-scratch/setup_path.sh                                    # Add dependencies to path
backends/cortex-m/test/setup_testing.sh                                   # Build executor-runner with cortex-m oplib + kernels registred
pytest --config-file=backends/arm/test/pytest.ini backends/cortex-m/test  # Run tests with correct configuration file
```

## Supported operators
Refer to `backends/cortex-m/test/ops` for currently supported accelerated ops/dtypes. Additionally, the quantizer targets pure "data-movement ops" such as data copies, slicing and concatinations to use quantized dtypes using the portable-kernels operator lbrary.
In general however, operators not supported by Cortex-M are kept in `fp32` using non-accelerated portable-kernels. It is recommended to analyze the graph after lowering to understand how much of the graph has been accelerated.

## Notices
Arm and Cortex are registered trademarks of Arm Limited (or its subsidiaries) in the US and/or elsewhere.
