# PyTorch::ExecuTorch CMSIS Pack

Build scripts and templates for the `PyTorch::ExecuTorch` CMSIS Pack.

## Overview

The CMSIS Pack packages the ExecuTorch runtime for Arm Cortex-M bare-metal
targets. Every portable and quantized operator is exposed as a selectable
CMSIS component, enabling fine-grained code-size control.

## Structure

```
backends/arm/
├── cmsis_pack/
│   ├── config/
│   │   └── executorch_config.yml       # Build configuration and defines
│   ├── contributions/
│   │   └── add/                        # Static files included in the pack
│   │       ├── LICENSE
│   │       └── Documentation/
│   ├── stubs/                          # Bare-metal runtime stubs
│   │   ├── bare_metal_pal.cpp
│   │   ├── cxx_runtime_stubs.cpp
│   │   ├── posix_stub.cpp
│   │   └── random_ops_stubs.cpp
│   └── templates/
│       └── PyTorch.ExecuTorch.pdsc.tpl # Pack description template
└── scripts/
    └── cmsis_pack/
        ├── build_pack.sh               # Main entry point
        ├── copy_sources.sh             # Collects sources from repo tree
        ├── generate_components.py      # Generates per-operator PDSC components
        └── generate_register_all_kernels.py  # Generates #ifdef-guarded registrations
```

## Components

- **Machine Learning::ExecuTorch::Runtime** — Core runtime (always required)
- **Machine Learning::ExecuTorch::Kernel Utils** — Kernel registration utilities
- **Machine Learning::ExecuTorch Operators::Portable \*** — Individual portable operators
- **Machine Learning::ExecuTorch Operators::Quantized \*** — Quantized operators
- **Machine Learning::ExecuTorch::Backend EthosU** — Ethos-U NPU backend for Cortex-M host (bare-metal)
- **Machine Learning::ExecuTorch::Backend EthosU Linux** — Ethos-U NPU backend for Cortex-A host (requires Linux userspace ethosu driver headers, supplied by the consumer project; mutually exclusive with the bare-metal variant)
- **Machine Learning::ExecuTorch::Backend CortexM** — CMSIS-NN optimized backend

## Building locally

```bash
# 1. Cross-compile ExecuTorch for Cortex-M (generates required headers)
cmake \
  -DCMAKE_TOOLCHAIN_FILE=examples/arm/ethos-u-setup/arm-none-eabi-gcc.cmake \
  -DEXECUTORCH_BUILD_ARM_BAREMETAL=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  -DEXECUTORCH_BUILD_FLATC=ON \
  -Bcmake-out-arm .
cmake --build cmake-out-arm --config Release -j$(nproc)

# 2. Build the pack
backends/arm/scripts/cmsis_pack/build_pack.sh \
  --executorch-root "$(pwd)" \
  --build-dir cmake-out-arm \
  --version "$(cat version.txt | sed 's/a0$//')" \
  --output-dir pack-output
```

The resulting `.pack` file is a zip archive installable via `cpackget add <file>.pack`.

## Dependencies

- ARM::CMSIS
- ARM::CMSIS-NN (for CMSIS-NN optimized operators)
- ARM::ethos-u-core-driver (for Ethos-U backend)
