# XNNPACK Backend

[XNNPACK](https://github.com/google/XNNPACK) is a library of optimized neural network operators for ARM and x86 CPU platforms. Our delegate lowers models to run using these highly optimized CPU operators. You can try out lowering and running some example models in the demo. Please refer to the following docs for information on the XNNPACK Delegate
- [XNNPACK Backend Delegate Overview](https://pytorch.org/executorch/stable/native-delegates-executorch-xnnpack-delegate.html)
- [XNNPACK Delegate Export Tutorial](https://pytorch.org/executorch/stable/tutorial-xnnpack-delegate-lowering.html)


## Directory structure

```bash
examples/xnnpack
├── quantization                      # Scripts to illustrate PyTorch 2 Export Quantization workflow with XNNPACKQuantizer
│   └── example.py
├── aot_compiler.py                   # The main script to illustrate the full AOT (export, quantization, delegation) workflow with XNNPACK delegate
├── xnn_executor_runner               # ExecuTorch runtime application for XNNPACK delegate examples
└── README.md                         # This file
```

## Delegating a Floating-point Model

The following command will produce a floating-point XNNPACK delegated model `mv2_xnnpack_fp32.pte` that can be run using XNNPACK's operators. It will also print out the lowered graph, showing what parts of the models have been lowered to XNNPACK via `executorch_call_delegate`.

```bash
# For MobileNet V2
python3 -m examples.xnnpack.aot_compiler --model_name="mv2" --delegate
```

Once we have the model binary (pte) file, then let's run it with ExecuTorch runtime using the `xnn_executor_runner`.

```bash
buck2 run examples/xnnpack:xnn_executor_runner -- --model_path ./mv2_xnnpack_fp32.pte
```

## Quantization
First, learn more about the generic PyTorch 2 Export Quantization workflow in the [Quantization Flow Docs](https://pytorch.org/executorch/stable/quantization-overview.html), if you are not familiar already.

Here we will discuss quantizing a model suitable for XNNPACK delegation using XNNPACKQuantizer.

Though it is typical to run this quantized mode via XNNPACK delegate, we want to highlight that this is just another quantization flavor, and we can run this quantized model without necessarily using XNNPACK delegate, but only using standard quantization operators.

A shared library to register the out variants of the quantized operators (e.g., `quantized_decomposed::add.out`) into EXIR is required. To generate this library, run the following command if using `buck2`:
```bash
buck2 build //kernels/quantized:aot_lib --show-output
```
Or if on cmake, follow the instructions in `test_quantize.sh` to build it, the default path is `cmake-out/kernels/quantized/libquantized_ops_lib.so`.

Then you can generate a XNNPACK quantized model with the following command by passing the path to the shared library into the script `quantization/example.py`:
```bash
python3 -m examples.xnnpack.quantization.example --model_name "mv2" --so_library "<path/to/so/lib>" # for MobileNetv2

# This should generate ./mv2_quantized.pte file, if successful.
```
You can find more valid quantized example models by running:
```bash
python3 -m examples.xnnpack.quantization.example --help
```

A quantized model can be run via `executor_runner`:
```bash
buck2 run examples/portable/executor_runner:executor_runner -- --model_path ./mv2_quantized.pte
```
Please note that running a quantized model will require the presence of various quantized/dequantize operators in the [quantized kernel lib](../../kernels/quantized).


## Delegating a Quantized Model

The following command will produce a XNNPACK quantized and delegated model `mv2_xnnpack_q8.pte` that can be run using XNNPACK's operators. It will also print out the lowered graph, showing what parts of the models have been lowered to XNNPACK via `executorch_call_delegate`.

```bash
python3 -m examples.xnnpack.aot_compiler --model_name "mv2" --quantize --delegate
```

Once we have the model binary (pte) file, then let's run it with ExecuTorch runtime using the `xnn_executor_runner`.

```bash
buck2 run examples/xnnpack:xnn_executor_runner -- --model_path ./mv2_xnnpack_q8.pte
```
