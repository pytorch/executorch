# Cadence DSP Backends

## Supported DSPs (in progress)
- HiFi Audio
- Fusion G3
- Vision P-Series

> **Note:** The Cadence DSP backends can only be compiled using the Xtensa toolchain (xt-clang cross-compiler). They cannot be built with standard host x86 compilers — the Xtensa toolchain is required for cross-compilation targeting any Cadence DSP family.

## Neural Network Libraries (nnlib)

Each DSP family uses a dedicated nnlib with optimized primitives:
- **HiFi**: [nnlib-hifi4](https://github.com/foss-xtensa/nnlib-hifi4)
- **Fusion G3**: [nnlib-FusionG3](https://github.com/foss-xtensa/nnlib-FusionG3/)

## Tutorial

Please follow the [tutorial](https://pytorch.org/executorch/main/backends-cadence) for more information on how to run models on Cadence/Xtensa DSPs.

## Directory Structure

```
executorch
├── backends
│   └── cadence
│       ├── aot
│       ├── generic
│       ├── utils
│       ├── hifi
│       │   ├── kernels
│       │   ├── operators
│       │   └── third-party
│       │       └── nnlib          # from nnlib-hifi4
│       ├── fusion_g3
│       │   ├── kernels
│       │   ├── operators
│       │   └── third-party
│       │       └── nnlib          # from nnlib-FusionG3
│       └── vision
│           ├── kernels
│           ├── operators
│           └── third-party
└── examples
    └── cadence
        ├── models
        └── operators
```
