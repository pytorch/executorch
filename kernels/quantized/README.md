# Quantized Ops

This folder contains kernels for quantization related ops, similar to `//executorch/kernels/portable/`.

## How to add quantized ops

1. Add a new operator definition in `quantized.yaml`

2. Implement the kernel for this operator.
3. Add unit test in `/test` directory.
4. Start to use `//executorch/kernels/quantized:generated_lib` in ExecuTorch or
    create your own generated lib if you only need a subset of the ops.
