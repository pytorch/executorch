# Custom Operator Registration Examples (WIP)
This folder contains examples to register custom operators into PyTorch as well as register its kernels into Executorch runtime.

## How to run

Prerequisite: finish the [setting up wiki](https://github.com/pytorch/executorch/blob/main/docs/website/docs/tutorials/00_setting_up_executorch.md).

Run:

```bash
bash test_custom_ops.sh
```

## AOT registration

In order to use custom ops in Executorch AOT flow (EXIR), the first option is to register the custom ops into PyTorch JIT runtime using `torch.library` APIs.

We can see the example in `custom_ops_1.py` where we try to register `my_ops::mul3` and `my_ops::mul3_out`. `my_ops` is the namespace and it will show up in the way we use the operator like `torch.ops.my_ops.mul3.default`. For more information about PyTorch operator, checkout [`pytorch/torch/_ops.py`](https://github.com/pytorch/pytorch/blob/main/torch/_ops.py).

Notice that we need both functional variant and out variant for custom ops, because EXIR will need to perform memory planning on the out variant `my_ops::mul3_out`.

## C++ kernel registration

After the model is exported by EXIR, we need C++ implementations of these custom ops in order to run it. `custom_ops_1.cpp` is an example C++ kernel. Other than that, we also need a way to bind the PyTorch op to this kernel. This binding is specified in `custom_ops.yaml`:
```yaml
- func: my_ops::mul3.out(Tensor input, *, Tensor(a!) output) -> Tensor(a!)
  kernels:
    - arg_meta: null
      kernel_name: custom::mul3_out_impl # sub-namespace native:: is auto-added
```
For how to write these YAML entries, please refer to [`kernels/portable/README.md`](https://github.com/pytorch/executorch/blob/main/kernels/portable/README.md).

Currently we provide 2 build systems that links `my_ops::mul3.out` kernel (written in `custom_ops_1.cpp`) to Executor runtime: buck2 and CMake. Both instructions are listed in `examples/custom_ops/test_custom_ops.sh`.
