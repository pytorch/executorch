# Qualcomm AI Engine Direct Backend

Disclaimer: At present, we do not offer any backward compatibility guarantees
for any APIs. We are currently in a pre-alpha development phase, and as such,
we reserve the right to modify interfaces and implementations.

This backend is implemented on the top of
[Qualcomm AI Engine Direct SDK](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk).
Please follow [setup](setup.md) to setup environment, build, and run executorch models by this backend.

## Delegate Options

Please check `generate_qnn_executorch_compiler_spec()` in
[utils.py](./utils/utils.py) for supported SoC and inference type.

### Supported Chipset
- Snapdragon 8 Gen 1
- Snapdragon 8 Gen 1+
- Snapdragon 8 Gen 2

### Supported Inference Type
- Quantized
- FP16

## Directory Structure

```
backends/qualcomm
├── aot # Codes for generating QNN context binary (AoT Part).
|   ├── wrappers # Wrapper of QNN data structures for ease of use.
|   └── python # Python interface for using QNN libraries.
├── builders # Codes for lowering each operators (AoT Part).
├── partition # QNN Partitioner (AoT Part).
├── passes # Various passes helping lower models to QNN backend (AoT Part).
├── python # Places to put pybind artifacts for accessing QNN APIs, structures, etc (AoT Part).
├── runtime # Here is QNN runtime responsbile for compiling a model on x64.
|   |       # Meanwhile, this is also the runtime responsbile for executing compiled
|   |       # models on a device.
|   └── backends # Backends supported by QNN.
|       └── htpbackend
|           ├── aarch64 # Configuration required to run on device. (Device Part).
|           └── x86_64 # Configuration required to compile graph on host. (AoT Part).
├── scripts # Misc supporting scripts, not related to core functionality.
├── tests # Unit tests and model tests go here.
└── utils # Miscellaneous utilities.

examples/backend
└── qualcomm # Examples to run QNN backends.
```

## Examples

Please see this [README.md](../../examples/backend/qualcomm/README.md).

Further, an example build script is provided as [build.sh](scripts/build.sh).

