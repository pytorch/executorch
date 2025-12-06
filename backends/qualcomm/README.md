# Qualcomm AI Engine Direct Backend

Disclaimer: At present, we do not offer any backward compatibility guarantees
for any APIs. We are currently in a development phase, and as such,
we reserve the right to modify interfaces and implementations.

This backend is implemented on the top of
[Qualcomm AI Engine Direct SDK](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk).
Please follow [tutorial](../../docs/source/backends-qualcomm.md) to setup environment, build, and run executorch models by this backend (Qualcomm AI Engine Direct is also referred to as QNN in the source and documentation).

A website version of the tutorial is [here](https://pytorch.org/executorch/main/backends-qualcomm).

## Delegate Options

Please check `generate_qnn_executorch_compiler_spec()` in
[utils.py](utils/utils.py) for supported SoC and inference type.

### Supported Chipset
- Snapdragon 8 Gen 1
- Snapdragon 8 Gen 1+
- Snapdragon 8 Gen 2
- Snapdragon 8 Gen 3
- Snapdragon 8 Elite
- Snapdragon 8 Elite Gen 5
- SA8295
- SA8255
- SSG2115P
- SSG2125P
- SXR1230P
- SXR2230P
- SXR2330P
- QCS9100
- SAR2230P

### Adding more supported Chipset
Currently, users cannot add additional chipset models because the chipset ID is not accessible to community users. If you have specific chipset models you wish to add, please contact one of the authors in the `Code Reviews` section at the bottom of this page.

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
├── _passes # Various private passes helping lower models to QNN backend (AoT Part).
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

## Issues
If you want to address the problem encountered, it would be great to have reproduction information for indicating maintainers. Please also follow the [policy](../../CONTRIBUTING.md#issues) to emit issues.

## Pull Requests
PRs are always welcome to help improve the codebase in a comprehensive manner. Before submitting changes, please apply:

- **Check the Coding Style**:<br/>
    Make sure your code follows the [style guides](../../CONTRIBUTING.md#coding-style) and passes the [lint checks](../../CONTRIBUTING.md#lintrunner).

- **Add Unit Tests**:<br/>
    Following is an example of adding test case after [creating new operator builder](builders/README.md), please navigate to `backends/qualcomm/tests` folder and put minimum example module in `model.py`. e.g.:
    ```python
    class IndexPut(torch.nn.Module):
        ...

    # please insert implementation in alphabetical order
    class LayerNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_norm = torch.nn.LayerNorm([768], eps=1e-6)

        def forward(self, x):
            return self.layer_norm(x)


    class LeakyReLUDefault(torch.nn.Module):
        ...
    ```
    Also extend sections `TestQNNFloatingPointOperator`, `TestQNNQuantizedOperator` in `test_qnn_delegate.py`. e.g.:
    ```python
    class TestQNNQuantizedOperator(TestQNN):
        def test_qnn_backend_interpolate_nearest_2d(self):
            ...

        # please insert it implementation alphabetical order
        def test_qnn_backend_layer_norm(self):
            module = LayerNorm()  # noqa: F405
            sample_input = (torch.randn(196, 768),)
            module = self.get_qdq_module(module, sample_input)
            self.lower_module_and_test_output(module, sample_input)

        def test_qnn_backend_leaky_relu(self):
            ...
    ```

- **Verify Unit Test Results**:<br/>
    ```bash
    cd $PATH_TO_EXECUTORCH
    # example usage of performing unit test
    python backends/qualcomm/tests/test_qnn_delegate.py -k TestQNNQuantizedOperator.test_qnn_backend_layer_norm -s $DEVICE_SERIAL -m SM8650 -b build-android/ -a $PATH_TO_TEST_ARTIFACTS
    ```
    The test graph is expected to have 1 delegated node with only placeholders / output nodes being left. Check the execution report for more information.

- **Code Reviews**:<br/>
    Please ping authors in Qualcomm AI Engine Direct related PRs for reviewing, possible candidates are listed below:
    - [shewu-quic](https://github.com/shewu-quic)
    - [chunit-quic](https://github.com/chunit-quic)
    - [winskuo-quic](https://github.com/winskuo-quic)
    - [DannyYuyang-quic](https://github.com/DannyYuyang-quic)
    - [haowhsu-quic](https://github.com/haowhsu-quic)

Thanks again for your contribution!
