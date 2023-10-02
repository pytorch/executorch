# Examples

This dir contains scripts and other helper utilities to illustrate an end-to-end workflow to run a torch.nn.module on the Executorch runtime.
It also includes a list of modules, from a simple `Add` to a full model like `MobileNetv2` and `MobileNetv3`, with more to come.


## Directory structure
```bash
examples
|── backend                           # Contains examples for exporting delegate models and running them using custom executor runners
├── custom_ops                        # Contains examples to register custom operators into PyTorch as well as register its kernels into ExecuTorch runtime
├── example_quantizer_and_delegate    # Contains examples to to fully lowered a MobileNetV2 model to the example backend with an example quantizer
├── export                            # Contains scripts to illustrate export workflow in portable mode
├── ios_demo_apps                     # Contains iOS demo apps
├── models                            # Contains a set of out-of-box PyTorch models
├── quantization                      # Contains examples of quantization workflow
├── recipes                           # Contains recipes for a set of demos
├── runtime                           # Contains examples of C++ wrapper around the ET runtime
└── README.md                         # This file
```

## Using the examples

We will walk through an example model to generate a binary file from a python torch.nn.module
from the `models` dir using scripts from the `export` dir. Then we will run on these binary
model files on the ExecuTorch (ET) runtime. For that we will use `executor_runner`. It is a simple
wrapper for the ExecuTorch runtime to serve as an example. Although simple, it is capable of loading
and executing previously exported binary file(s).


1. Following the setup guide in [Setting up ExecuTorch from GitHub](/docs/website/docs/tutorials/00_setting_up_executorch.md)
you should be able to get the basic development environment for ExecuTorch working.

2. Using the example script `portable/aot_compiler.py` generate a model binary file by selecting a
model name from the list of available models in the `models` dir.


```bash
cd executorch # To the top level dir

# To get a list of example models
python3 -m examples.export.portable -h

# To generate a specific pte model
python3 -m examples.export.portable --model_name="mv2" # for MobileNetv2

# This should generate ./mv2.pte file, if successful.
```

Use `-h` (or `--help`) to see all the supported models.

3. Once we have the model binary (pte) file, then let's run it with ExecuTorch runtime using the `executor_runner`.

```bash
buck2 run examples/runtime/portable:executor_runner -- --model_path mv2.pte
```

## Quantization
Here is the [Quantization Flow Docs](/docs/website/docs/tutorials/quantization_flow.md).

### Generating quantized model

You can generate quantized model with the following command (following example is for mv2, aka MobileNetV2):
```bash
python3 -m examples.quantization.example --model_name "mv2" --so-library "<path/to/so/lib>" # for MobileNetv2
```

Note that the shared library being passed into `example.py` is required to register the out variants of the quantized operators (e.g., `quantized_decomposed::add.out`)into EXIR. To build this library, run the following command if using buck2:
```bash
buck2 build //kernels/quantized:aot_lib --show-output
```

If on cmake, follow the instructions in `test_quantize.sh` to build it, the default path is `cmake-out/kernels/quantized/libquantized_ops_lib.so`.

This command will print both the original model after capture and quantized model.

The flow produces a quantized model that could be lowered through partitioner or the runtime directly.


you can also find the valid quantized example models by running:
```bash
buck2 run executorch/examples/quantization:example -- --help
```

### Running quantized model

Quantized model can be run via executor_runner, similar to floating point model, via, as shown above:

```bash
buck2 run examples/runtime/portable:executor_runner -- --model_path mv2.pte
```

Note that, running quantized model, requires various quantized/dequantize operators, available in [quantized kernel lib](/kernels/quantized).

## XNNPACK Backend
Please see [Backend README](backend/README.md) for XNNPACK quantization, export, and run workflow.

## Dependencies

Various models listed in this directory have dependencies on some other packages, e.g. torchvision, torchaudio.
In order to make sure model's listed in examples are importable, e.g. via

```python
from executorch.examples.models.mobilenet_v3d import MV3Model
m = MV3Model.get_model()
```
You need to follow the setup guide in [Setting up ExecuTorch from GitHub](/docs/website/docs/tutorials/00_setting_up_executorch.md) to have appropriate packages installed. If you haven't already, install these deps via

```bash
cd executorch

bash ./install_requirements.sh
```
