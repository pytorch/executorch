# Qualcomm AI Engine Direct Backend

Disclaimer: At present, we do not offer any backward compatibility guarantees
for any APIs. We are currently in a pre-alpha development phase, and as such,
we reserve the right to modify interfaces and implementations.

This backend is implemented on the top of
[Qualcomm AI Engine Direct SDK](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk).
Please follow [setup](setup.md) to setup environment, build, and run executorch models by this backend.

# Delegate options

Please check `generate_qnn_executorch_compiler_spec()` in
[utils.py](backends/qualcomm/utils/utils.py) for supported SoC and inference type.

### supported chipset
- Snapdragon 8 Gen 1
- Snapdragon 8 Gen 1+
- Snapdragon 8 Gen 2

### supported inference type
- Quantized
- FP16

# Directory Structure

```
backends/qualcomm
├── builders # Codes for lowering each operators.
├── examples # Examples to run QNN backends.
├── partition # QNN Partitioner.
├── passes # Various passes helping lower models to QNN backend.
├── python # Places to put pybind artifacts for accessing QNN APIs, structures, etc.
├── runtime # Here is QNN runtime responsbile for compiling a model on x64.
            # Meanwhile, this is also the runtime responsbile for executing compiled
            # models on a device.
├── scripts # Misc supporting scripts, not related to core functionality.
├── tests # Unit tests and model tests go here.
└── utils # Miscellaneous utilities.
```

# Examples

Please see this [README.md](../../examples/backend/qualcomm/README.md).

Further, an example build script is provided as [build.sh](scripts/build.sh).

