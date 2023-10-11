# XNNPACK Backend

[XNNPACK](https://github.com/google/XNNPACK) is a library of optimized of neural network inference operators for ARM and x86 platforms. Our delegate lowers models to run using these highly optimized CPU operators. You can try out lowering and running some example models in the demo. Please refer to the following docs for information on the XNNPACK Delegate
- [XNNPACK Backend Delegate Overview](https://github.com/pytorch/executorch/blob/main/docs/website/docs/source/native-delegates-executorch-xnnpack-delegate.md)
- [XNNPACK Delegate Export Tutorial](https://github.com/pytorch/executorch/blob/main/docs/website/docs/source/tutorial-xnnpack-delegate-lowering.md)


## Directory structure
```bash
examples/xnnpack
├── quantization                      # Scripts to illustrate PyTorch 2.0 quantization workflow with XNNPACK quantizer
│   └── example.py
├── aot_compiler.py                   # The main script to illustrate the full AOT (export, quantization, delegation) workflow with XNNPACK
├── xnn_executor_runner               # ExecuTorch runtime with XNNPACK
└── README.md                         # This file
```

## XNNPACK delegation-only

The following command will produce a floating-point XNNPACK delegated model `mv2_xnnpack_fp32.pte` that can be run using XNNPACK's operators. It will also print out the lowered graph, showing what parts of the models have been lowered to XNNPACK via `executorch_call_delegate`.

```bash
# For MobileNet V2
python3 -m examples.xnnpack.aot_compiler --model_name="mv2" --delegate
```

Once we have the model binary (pte) file, then let's run it with ExecuTorch runtime using the `xnn_executor_runner`.

```bash
buck2 run examples/xnnpack:xnn_executor_runner -- --model_path ./mv2_xnnpack_fp32.pte
```

## XNNPACK quantization + delegation

The following command will produce a XNNPACK quantized and delegated model `mv2_xnnpack_q8.pte` that can be run using XNNPACK's operators. It will also print out the lowered graph, showing what parts of the models have been lowered to XNNPACK via `executorch_call_delegate`.

```bash
python3 -m examples.xnnpack.aot_compiler --model_name "mv2" --quantize --delegate
```

Once we have the model binary (pte) file, then let's run it with ExecuTorch runtime using the `xnn_executor_runner`.

```bash
buck2 run examples/xnnpack:xnn_executor_runner -- --model_path ./mv2_xnnpack_q8.pte
```

## XNNPACK quantization
Learn the generic PyTorch 2.0 quantization workflow in the [Quantization Flow Docs](/docs/website/docs/tutorials/quantization_flow.md).


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
buck2 run executorch/examples/xnnpack/quantization:example -- -h
```

A quantized model can be run via `executor_runner`:
```bash
buck2 run examples/portable/executor_runner:executor_runner -- --model_path ./mv2_quantized.pte
```
Please note that running a quantized model will require the presence of various quantized/dequantize operators in the [quantized kernel lib](../../kernels/quantized).
