# Selective Build Examples
To optimize binary size of ExecuTorch runtime, selective build can be used. This folder contains examples to select only the operators needed for Executorch build. We provide APIs for both CMake build and buck2 build. This example will demonstrate both. You can find more information on how to use buck2 macros in [wiki](https://github.com/pytorch/executorch/blob/main/docs/website/docs/tutorials/selective_build.md).

## How to run

Prerequisite: finish the [setting up wiki](https://github.com/pytorch/executorch/blob/main/docs/website/docs/tutorials/00_setting_up_executorch.md).

Run:

```bash
bash test_selective_build.sh
```

## BUCK2 examples

Check out `targets.bzl` for demo of 3 selective build APIs:
1. Select all ops from the dependency kernel libraries, register all of them into Executorch runtime.
2. Only select ops from `ops` kwarg in `et_operator_library` macro.
3. Only select from a yaml file from `ops_schema_yaml_target` kwarg in `et_operator_library` macro.

We have one more API incoming: only select from an exported model file (.pte).

## CMake examples

TODO
