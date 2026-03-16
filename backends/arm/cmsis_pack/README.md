# PyTorch::ExecuTorch CMSIS Pack

This directory contains the build scripts and templates to generate the `PyTorch::ExecuTorch` CMSIS Pack.

## Overview

The ExecuTorch CMSIS Pack provides the ExecuTorch runtime library for embedded systems with all operators exposed as selectable components. This enables fine-grained dependency management where model packs can select only the operators they need.

## Structure

```
executorch-pack/
├── build/                  # Generated pack build output
├── templates/
│   └── PyTorch.ExecuTorch.pdsc.tpl    # Pack description template
├── scripts/
│   ├── build_pack.sh       # Main pack build script
│   └── generate_components.py  # Operator component generator
├── config/
│   └── executorch_config.yml   # Build configuration and defines
└── contributions/
    └── add/                # Static files to include in pack
        ├── LICENSE
        └── Documentation/
```

## Components

The pack provides the following component hierarchy:

- **Machine Learning::ExecuTorch::Runtime** - Core runtime (always required)
- **Machine Learning::ExecuTorch::Kernel Utils** - Kernel registration infrastructure
- **Machine Learning::ExecuTorch::Operators::Portable::*** - Individual portable operators
- **Machine Learning::ExecuTorch::Operators::Quantized::*** - Quantized operators
- **Machine Learning::ExecuTorch::Backend::EthosU** - Ethos-U NPU backend (optional)
- **Machine Learning::ExecuTorch::Backend::CortexM** - Cortex-M optimized backend (optional)

## Building

The pack is built using a Docker container. Build the image from the [cmsis-executorch](https://github.com/Arm-Examples/cmsis-executorch) Dockerfile:

```bash
# Build Docker image locally from remote Dockerfile
curl -sL https://raw.githubusercontent.com/Arm-Examples/CMSIS-Executorch/main/.docker/Dockerfile | docker build -t executorch-arm-container:latest -

# From repository root, run the build script
docker run --rm -v $(pwd):/workspace2 executorch-arm-container:latest /workspace2/scripts/build_pack.sh
```

**Build Versioning**: Each pack build automatically increments a build number using SemVer pre-release format: `MAJOR.MINOR.PATCH-build.BUILD` (e.g., `1.0.0-build.42`). See [BUILD_NUMBER.md](BUILD_NUMBER.md) for details.

## Dependencies

- ARM::CMSIS
- ARM::CMSIS-NN (for CMSIS-NN optimized operators)
- ARM::ethos-u-core-driver (for Ethos-U backend)
