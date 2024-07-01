# Selective Build Examples
To optimize binary size of ExecuTorch runtime, selective build can be used. This folder contains examples to select only the operators needed for ExecuTorch build. We provide APIs for both CMake build and buck2 build. This example will demonstrate the CMake build. To use the buck2 API, check `test_selective_build.sh` in the current directory. You can find more information on how to use buck2 macros in [wiki](../../docs/source/kernel-library-selective_build.md).

## How to run

Prerequisite: finish the [setting up wiki](https://pytorch.org/executorch/stable/getting-started-setup).

Run:

```bash
cd executorch
bash examples/selective_build/test_selective_build.sh cmake
```

Check out `CMakeLists.txt` for demo of 3 selective build APIs:
1. `SELECT_ALL_OPS`: Select all ops from the dependency kernel libraries, register all of them into ExecuTorch runtime.
2. `SELECT_OPS_LIST`: Only select from a list of ops.
3. `SELECT_OPS_YAML`: Only select from a yaml file.

Other configs:
- `MAX_KERNEL_NUM=N`: Only allocate memory for N operators.

We have one more API incoming: only select from an exported model file (.pte).
