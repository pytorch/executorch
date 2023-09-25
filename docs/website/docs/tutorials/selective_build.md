# Selective Build



## Introduction

**Warning: Selective build process is required by all ExecuTorch operator libraries! Not having it properly setup will cause the operator to not being registered.**

Selective build helps reduce ExecuTorch build binary size, improving code structure by avoiding duplicate ATen-compliant operator schemas and definitions. It should be the by default build mode for ExecuTorch builds.

During development when binary size is not an issue, the option with a full list of operators is also provided (refer to `include_all_ops=True` below in detail).

## How does it work

On a high level, scripts under `codegen/*` extract out operators being used by a model (or multiple models) and write the information into a yaml file. Codegen system reads this yaml file and selectively generates C++ code to register the corresponding operators and code to call the kernels. All the generated files will be encapsulated into a BUCK (or TARGETS) target, which needs to depend on the kernel libraries. Then both generated library and kernel libraries will be included into this ExecuTorch build.



## APIs



### et_operator_library

This is very much similar to PyTorch mobile selective build rule [`pt_operator_library`](https://fburl.com/code/i8wmbuq2). This rule takes 4 types of inputs:

1. a target pointing to a model file

For example, `//custom/models:model_1` is the target that points to a checked-in model file, we can write a rule similar as:
```python
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "et_operator_library")

fb_native.export_file(
  name = "model_1",
  src = "model_1.pte", # checked in model
)

et_operator_library(
  name = "selective_ops",
  model = [
    "//custom/models:model_1",
  ],
)
```
Under the hood we will generate a `model_operators.yaml` file for this model.

2. a list of operators in plain text

For example, if we want to include `aten::add` and `aten::mul` into ExecuTorch build, we can write a rule like:
```python
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "et_operator_library")

et_operator_library(
  name = "selective_ops",
  ops = [
    "aten::add",
    "aten::mul",
  ],
)
```
3. a `functions.yaml` file or a `custom_ops.yaml` file. Pass the yaml file target into `et_operator_libary` and be done with it.
```python
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "et_operator_library")

export_file(
  name = "functions.yaml"
)

et_operator_library(
  name = "selective_ops",
  ops_schema_yaml_path = ":functions.yaml",
)
```
4. a boolean indicates that we want to include all ops.
If we want to include all the operators listed in `functions.yaml` as well as `custom_ops.yaml` , we can simply do:
```python
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "et_operator_library")

et_operator_library(
  name = "selective_ops",
  include_all_ops = True,
)
```
### executorch_generated_lib

After setting up selected operators by using `et_operator_library` rule, user can let a `executorch_generated_lib` to be depending on them. Notice that one `executorch_generated_lib` can depend on multiple `et_operator_library`, and internally we will union all the selected operators by aggregating all the `model_operators.yaml` files into a `selective_operators.yaml`. This `selective_operators.yaml` will be consumed by codegen system later.

## Example
To show an example, say we have these two op libraries:

```python
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "et_operator_library")

et_operator_library(
  name = "selective_ops_list",
  ops = [
    "aten::add",
    "aten::mul",
  ],
)

et_operator_library(
  name = "selective_ops_model",
  model = [
    "//custom/models:model_1", # say it contains "aten::div"
  ],
)
```
Our `executorch_generated_lib` needs to be depending on both of them:
```python
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "executorch_generated_lib")

executorch_generated_lib(
  name = "custom_generated_lib`,
  deps = [
    ":selective_ops_list",
    ":selective_ops_model",
  ],
)
```
Then we will be able to generate code for all 3 operators: `aten::add`, `aten::mul` and `aten::div`.
