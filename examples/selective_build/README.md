# Selective Build Examples
To optimize binary size of ExecuTorch runtime, selective build can be used. This folder contains examples to select only the operators needed for ExecuTorch build. This example will demonstrate the CMake build.

## How to run

Prerequisite: finish the [setting up wiki](https://pytorch.org/executorch/stable/getting-started-setup).

Run:

```bash
cd executorch
bash examples/selective_build/test_selective_build.sh cmake
```

Check out `CMakeLists.txt` for demo of 3 selective build APIs:
1. `SELECT_ALL_OPS`: Select all ops from the dependency kernel libraries, register all of them into ExecuTorch runtime.
2. `SELECT_OPS_LIST`: Only select operators from a list.
3. `SELECT_OPS_YAML`: Only select operators from a yaml file.

Other configs:
- `MAX_KERNEL_NUM=N`: Only allocate memory for N operators.

We have one more API incoming: only select from an exported model file (.pte).
