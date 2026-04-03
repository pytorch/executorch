# Cortex-M Backend

> [!NOTE]
> Beta. This backend has been validated with a set of small models (e.g. MLPerf Tiny, MobileNetV2) and provides broad operator coverage through CMSIS-NN accelerated kernels with portable-ops fallback.

## Overview

The Cortex-M backend is implemented as an operator dialect/library based on [CMSIS-NN](https://github.com/ARM-software/CMSIS-NN), together with the `CortexMQuantizer` which targets supported ops, and the `CortexMPassManager` which modifies the exported program to use Cortex-M operators where possible. It is intended for use with **channels-last input**  since this is what the accelerated kernels are using.

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

## Supported operators
Refer to `backends/cortex_m/test/ops` for currently supported accelerated ops/dtypes. Additionally, the quantizer targets pure "data-movement ops" such as data copies, slicing and concatenations to use quantized dtypes using the portable-kernels operator library.
In general however, operators not supported by Cortex-M are kept in `fp32` using non-accelerated portable-kernels. It is recommended to analyze the graph after lowering to understand how much of the graph has been accelerated.

## Notices
Arm and Cortex are registered trademarks of Arm Limited (or its subsidiaries) in the US and/or elsewhere.
