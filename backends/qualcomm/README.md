# Qualcomm AI Engine Direct Backend

Disclaimer: At present, we do not offer any backward compatibility guarantees
for any APIs. We are currently in a development phase, and as such,
we reserve the right to modify interfaces and implementations.

This backend is implemented on the top of
[Qualcomm AI Engine Direct SDK](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk).
Please follow [tutorial](../../docs/source/build-run-qualcomm-ai-engine-direct-backend.md) to setup environment, build, and run executorch models by this backend (Qualcomm AI Engine Direct is also referred to as QNN in the source and documentation).

A website version of the tutorial is [here](https://pytorch.org/executorch/stable/build-run-qualcomm-ai-engine-direct-backend.html).

## Delegate Options

Please check `generate_qnn_executorch_compiler_spec()` in
[utils.py](./utils/utils.py) for supported SoC and inference type.

### Supported Chipset
- Snapdragon 8 Gen 1
- Snapdragon 8 Gen 1+
- Snapdragon 8 Gen 2
- Snapdragon 8 Gen 3

### How to add more supported Chipset

#### Step 1: Check SoC model of snapdragon device
Get SoC model which would like to be supported from the document of Qualcomm AI Engine Direct SDK.

#### Step 2: Update schema of compiler option and SoC information in serialization
Add SoC model into QcomChipset enum in [schema](./serialization/schema.fbs) and [qnn_compile_spec_schema](./serialization/qnn_compile_spec_schema.py).
Insert new SoC information into _soc_info_table in [qnn_compile_spec_schema](./serialization/qnn_compile_spec_schema.py).

#### Step 3: Recompile the .pte file
Follow [setup](../../docs/source/build-run-qualcomm-ai-engine-direct-backend.md) to setup environment and build runtime with new schema header.

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
├── quantizer # QNN Quantizer
├── runtime # Here is QNN runtime responsbile for compiling a model on x64.
|   |       # Meanwhile, this is also the runtime responsbile for executing compiled
|   |       # models on a device.
|   └── backends # Backends supported by QNN.
|       └── htpbackend
|           ├── aarch64 # Configuration required to run on device. (Device Part).
|           └── x86_64 # Configuration required to compile graph on host. (AoT Part).
├── scripts # Misc supporting scripts, not related to core functionality.
├── serialization # Contains files related to serializing QNN compiler options and SoC information
├── tests # Unit tests and model tests go here.
└── utils # Miscellaneous utilities.

examples/qualcomm
├── executor_runner # A general runner that is capable of running most of the basic models.
├── oss_scripts # Scripts for OSS(Open Source Software) models and customized runner for some specific models.
├── qaihub_scripts # Scripts for Qaihub models and corresponding customized runner for these models.
└── scripts # Scripts for models provided by executorch.
```

## Examples

Please see this [README.md](../../examples/qualcomm/README.md).

Further, an example build script is provided as [build.sh](scripts/build.sh).
