# Selective Build Examples
To optimize binary size of ExecuTorch runtime, selective build can be used. This folder contains examples to select only the operators needed for ExecuTorch build. This example will demonstrate the CMake build.

## How to run

Prerequisite: finish the [setting up wiki](https://pytorch.org/executorch/main/getting-started-setup).

Run:

```bash
cd executorch
bash examples/selective_build/test_selective_build.sh cmake
```

Check out `CMakeLists.txt` for demo of selective build APIs:
1. `SELECT_ALL_OPS`: Select all ops from the dependency kernel libraries, register all of them into ExecuTorch runtime.
2. `SELECT_OPS_LIST`: Only select operators from a list.
3. `SELECT_OPS_YAML`: Only select operators from a yaml file.
4. `SELECT_OPS_FROM_MODEL`: Only select operators from a from an exported model pte.
5. `DTYPE_SELECTIVE_BUILD`: Enable rebuild of `portable_kernels` to use dtype selection. Currently only supported for `SELECTED_OPS_FROM_MODEL` API and `portable_kernels` lib.

Other configs:
- `MAX_KERNEL_NUM=N`: Only allocate memory for N operators.
