# Custom Operator Registration Examples
This folder contains examples to register custom operators into PyTorch as well as register its kernels into ExecuTorch runtime.

## How to run

Prerequisite: finish the [setting up wiki](https://pytorch.org/executorch/stable/getting-started-setup).

Run:

```bash
cd executorch
bash examples/portable/custom_ops/test_custom_ops.sh
```

## AOT registration

In order to use custom ops in ExecuTorch AOT flow (EXIR), the first option is to register the custom ops into PyTorch JIT runtime using `torch.library` APIs.

We can see the example in `custom_ops_1.py` where we try to register `my_ops::mul3` and `my_ops::mul3_out`. `my_ops` is the namespace and it will show up in the way we use the operator like `torch.ops.my_ops.mul3.default`. For more information about PyTorch operator, checkout [`pytorch/torch/_ops.py`](https://github.com/pytorch/pytorch/blob/main/torch/_ops.py).

Notice that we need both functional variant and out variant for custom ops, because EXIR will need to perform memory planning on the out variant `my_ops::mul3_out`.

The second option is to register the custom ops into PyTorch JIT runtime using C++ APIs (`TORCH_LIBRARY`/`TORCH_LIBRARY_IMPL`). This also means we need to write C++ code and it needs to depend on `libtorch`.

We added an example in `custom_ops_2.cpp` where we implement and register `my_ops::mul4`, also `custom_ops_2_out.cpp` with an implementation for `my_ops::mul4_out`.

By linking them both with `libtorch` and `executorch` library, we can build a shared library `libcustom_ops_aot_lib_2` that can be dynamically loaded by Python environment and then register these ops into PyTorch. This is done by `torch.ops.load_library(<path_to_libcustom_ops_aot_lib_2>)` in `custom_ops_2.py`.

## C++ kernel registration

After the model is exported by EXIR, we need C++ implementations of these custom ops in order to run it. For example, `custom_ops_1_out.cpp` is a C++ kernel that can be plugged into the ExecuTorch runtime. Other than that, we also need a way to bind the PyTorch op to this kernel. This binding is specified in `custom_ops.yaml`:
```yaml
- func: my_ops::mul3.out(Tensor input, *, Tensor(a!) output) -> Tensor(a!)
  kernels:
    - arg_meta: null
      kernel_name: custom::mul3_out_impl # sub-namespace native:: is auto-added
```
For how to write these YAML entries, please refer to [`kernels/portable/README.md`](https://github.com/pytorch/executorch/blob/main/kernels/portable/README.md).

Currently we use Cmake as the build system to link the `my_ops::mul3.out` kernel (written in `custom_ops_1.cpp`) to the ExecuTorch runtime. See instructions in: `examples/portable/custom_ops/test_custom_ops.sh` (test_cmake_custom_op_1).

## Selective build

Note that we have defined a custom op for both `my_ops::mul3.out` and `my_ops::mul4.out` in `custom_ops.yaml`. To reduce binary size, we can choose to only register the operators used in the model. This is done by passing in a list of operators to the `gen_oplist` custom rule, for example: `--root_ops="my_ops::mul4.out"`.

We then let the custom ops library depend on this target, to only register the ops we want.

For more information about selective build, please refer to [`selective_build.md`](../../../docs/source/kernel-library-selective-build.md).
