# Cadence DSP Backends

## Supported DSPs (in progress)
- HiFi Audio
- ...

## Tutorial

Please follow the [tutorial](https://pytorch.org/executorch/main/build-run-xtensa.html) for more information on how to run models on Cadence/Xtensa DSPs.

## Directory Structure

```
executorch
├── backends
│   └── cadence
│       ├── aot
│       ├── ops_registration
│       ├── tests
│       ├── utils
│       └── hifi
│           ├── kernels
│           ├── operators
│           └── third-party
│               └── nnlib
└── examples
    └── cadence
        ├── models
        └── operators
```
