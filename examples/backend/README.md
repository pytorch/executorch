This README gives some examples on backend-specific model workflow.

# XNNPACK Backend

[XNNPACK](https://github.com/google/XNNPACK) is a library of optimized of neural network inference operators for ARM and x86 platforms. Our delegate lowers models to run using these highly optimized CPU operators. You can try out lowering and running some example models using the following commands:

## XNNPACK delegation-only

The following command will produce an floating-point XNNPACK delegated model `mv2_xnnpack_fp32.pte` that can be run using XNNPACK's operators. It will also print out the lowered graph, showing what parts of the models have been lowered to XNNPACK via `executorch_call_delegate`.

```bash
# For MobileNet V2
python3 -m examples.backend.xnnpack_examples --model_name="mv2" --delegate
```

Once we have the model binary (pte) file, then let's run it with Executorch runtime using the `xnn_executor_runner`.

```bash
buck2 run examples/backend:xnn_executor_runner -- --model_path ./mv2_xnnpack_fp32.pte
```

## XNNPACK quantization + delegation
The following command will produce an XNNPACK quantized and delegated model `mv2_xnnpack_q8.pte` that can be run using XNNPACK's operators. It will also print out the lowered graph, showing what parts of the models have been lowered to XNNPACK via `executorch_call_delegate`.

```bash
python3 -m examples.backend.xnnpack_examples --model_name="mv2" --quantize --delegate
```

Once we have the model binary (pte) file, then let's run it with Executorch runtime using the `xnn_executor_runner`.

```bash
buck2 run examples/backend:xnn_executor_runner -- --model_path ./mv2_xnnpack_q8.pte
```
