# Qualcomm AI Engine Direct Backend

Disclaimer: At present, we do not offer any backward compatibility guarantees
for any APIs. We are currently in a pre-alpha development phase, and as such,
we reserve the right to modify interfaces and implementations.

This backend is implemented on the top of
[Qualcomm AI Engine Direct SDK](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk).
Please follow [setup](setup.md) to setup environment, build, and run executorch models by this backend.

# Delegate options

Below is short descriptions for available options introduced by
`generate_qnn_executorch_compiler_spec()`.

* `is_fp16`: If true, the model is compiled to QNN HTP fp16 runtime.
Note that not all SoC support QNN HTP fp16. Only premium tier SoC like
Snapdragon 8 Gen 1 or newer can support HTP fp16.

* `soc_model`: The SoC you plan to run the compiled model. Please check
`generate_qnn_executorch_compiler_spec()` in
[utils.py](backends/qnn/utils/utils.py) for supported SoC. At the moment of
writting this README, `SM8450`(Snapdragon 8 Gen 1), `SM8475`(Snapdragon 8 Gen 1+),
`SM8550`(Snapdragon 8 Gen 2) are supported.

* `debug`: Enable verbose logging. Disclaimer: this option must change in the
near future.

* `saver`: Instead of compiling the model, run QNN Saver. Please check
documents of Qualcomm AI Engine Direct SDK. This feature is usually for debugging
purpose.

# Directory Structure

```
backends/qnn
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

Please see [examples/README.md](examples/README.md).

Further, an example build script is provided as [build.sh](scripts/build.sh).

