# Selective Build Examples
To optimize binary size of ExecuTorch runtime, selective build can be used. This folder contains examples to select only the operators needed for ExecuTorch build. We provide APIs for both CMake build and buck2 build. This example will demonstrate both. You can find more information on how to use buck2 macros in [wiki](../../docs/source/kernel-library-selective_build.md).

## How to run

Prerequisite: finish the [setting up wiki](https://pytorch.org/executorch/stable/getting-started-setup).

Run:

```bash
cd executorch
bash examples/selective_build/test_selective_build.sh [cmake|buck2]
```

## BUCK2 examples

Check out `targets.bzl` for demo of 4 selective build APIs:
1. `--config executorch.select_ops=all`: Select all ops from the dependency kernel libraries, register all of them into ExecuTorch runtime.
2. `--config executorch.select_ops=list`: Only select ops from `ops` kwarg in `et_operator_library` macro.
3. `--config executorch.select_ops=yaml`: Only select from a yaml file from `ops_schema_yaml_target` kwarg in `et_operator_library` macro.
4. `--config executorch.select_ops=dict`: Only select ops with from `ops_dict` kwarg in `et_operator_library` macro. Optionally, add dtype information to each operator and add `dtype_selective_build = True` to only select those dtypes.
Eg. if the model only uses the float implementation of add, then only the float add will be registered. Note: setting `dtype_selective_build = True` is only available in xplat.

Other configs:
- `--config executorch.max_kernel_num=N`: Only allocate memory for the required number of operators. Take this result from `selected_operators.yaml`.

## CMake examples

Check out `CMakeLists.txt` for demo of 3 selective build APIs:
1. `SELECT_ALL_OPS`
2. `SELECT_OPS_LIST`
3. `SELECT_OPS_YAML`

Other configs:
- `MAX_KERNEL_NUM=N`

We have one more API incoming: only select from an exported model file (.pte).
