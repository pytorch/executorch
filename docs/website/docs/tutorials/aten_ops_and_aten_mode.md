# ATen-compliant Operator Registration & ATen mode


## Introduction

ExecuTorch supports a subset of ATen-compliant operators.
ATen-compliant operators are those defined in
[`native_functions.yaml`](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml),
with their native functions (or kernels, we use these two terms interchangeably)
either defined in ATen library or other user defined libraries. The ATen-compliant operators supported by ExecuTorch have these traits (actually same for custom ops):
1. Out variant, means these ops take an `out` argument
2. Functional except `out`. These ops shouldn't mutate input tensors other than `out`, shouldn't create aliasing views.

To give an example, `aten::add_.Tensor` is not supported since it mutates an input tensor, `aten::add.out` is supported.

ATen mode is a build-time option to link ATen library into ExecuTorch runtime, so those registered ATen-compliant ops can use their original ATen kernels.

On the other hand we need to provide our custom kernels if ATen mode is off (a.k.a. lean mode).

In the next section we will walk through the steps to register ATen-compliant ops into ExecuTorch runtime.

## Step by step guide
There are two branches for this use case:
* ATen mode. In this case we expect the exported model to be able to run with ATen kernels .
* Lean mode. This requires ATen-compliant op implementations using `ETensor`.

In a nutshell, we need the following steps in order for a ATen-compliant op to work on ExecuTorch:

#### ATen mode:
1. Define a target for selective build (`et_operator_library` macro)
2. Pass this target to codegen using `executorch_generated_lib` macro
3. Hookup the generated lib into ExecuTorch runtime.

For more details on how to use selective build, check [Selective Build](https://www.internalfb.com/intern/staticdocs/executorch/docs/tutorials/custom_ops/#selective-build).
#### Lean mode:
1. Declare the op name in `functions.yaml`. Detail instruction can be found in [Declare the operator in a YAML file](https://www.internalfb.com/code/fbsource/xplat/executorch/kernels/portable/README.md).
2. (not required if using ATen mode) Implement the kernel for your operator using `ETensor`. ExecuTorch provides a portable library for frequently used ATen-compliant ops. Check if the op you need is already there, or you can write your own kernel.
3. Specify the kernel namespace and function name in `functions.yaml` so codegen knows how to bind operator to its kernel.
4. Let codegen machinery generate code for either ATen mode or lean mode, and hookup the generated lib into ExecuTorch runtime.

### Case Study
Let's say a model uses an ATen-compliant operator `aten::add.out`.

We can either reuse the kernel written in portable library [here](https://www.internalfb.com/code/fbsource/xplat/executorch/kernels/portable/cpu/op_add.cpp) or choose to write our own.

#### ATen mode

For ATen mode we don't need to define any `functions.yaml` file or write any kernel, since they are already defined in `native_functions.yaml`. All we need to do is to use selective build to choose the ops we want:
```python
et_operator_library(
  name = "selected_ops",
  ops = [
    "aten::add.out",
  ],
)

executorch_generated_lib(
  name = "add_lib",
  deps = [
    ":selected_ops",
  ],
  aten_mode = True,
)
```

#### Lean mode

Let's say we like to write our own kernel:

```cpp
// add.cpp
namespace custom {
namespace native {

Tensor& add_out(const Tensor& a, const Tensor& b, const Scalar& alpha, Tensor& out) {
  // do something
  return out;
}

} // namespace native
} // namespace custom
```

The corresponding `functions.yaml` for this operator looks like:

```yaml
- op: add.out
  dispatch:
    CPU: custom::add_out
```
Notice that there are some caveats:
#### Caveats
* `dispatch` and `CPU` are legacy fields and they don't mean anything in ExecuTorch context.
* Namespace `aten` is omitted.
* We don't need to write `aten::add.out` function schema  because we will use the schema definition in `native_functions.yaml` as our source of truth.
* Kernel namespace in the yaml file is `custom` instead of `custom::native`. This is because codegen will append a `native` namespace automatically. It also means the kernel always needs to be defined under `<name>::native`.

Now we need to trigger codegen to generate operator library:
```python
export_file(
  name = "functions.yaml"
)

cxx_library(
  name = "add_kernel",
  srcs = ["add.cpp"],
)

et_operator_library(
  name = "all_ops",
  include_all_ops = True, # Select all ops in functions.yaml
)

executorch_generated_lib(
  name = "add_lib",
  functions_yaml_target = ":functions.yaml",
  deps = [
    ":all_ops",
    ":add_kernel",
  ],
)
```
### Usage of generated lib
In the case study above, eventually we have `add_lib` which is a C++ library responsible to register `aten::add.out` into ExecuTorch runtime.

In our ExecuTorch binary target, add `add_lib` as a dependency:
```python
cxx_binary(
  name = "executorch_bin",
  deps = [
    "//executorch/runtime/executor:program", # Program and Method
    "//path/to:add_lib", # operator library
  ],
)
```
## Common APIs

To facilitate custom operator registration, we provide the following APIs:

- `functions.yaml`: ATen-compliant operator schema and kernel metadata are defined in this file.
- `executorch_generated_lib`: the Buck rule to call ExecuTorch codegen system and encapsulate generated C++ source files into libraries. If only include ATen-compliant operators, only one library will be generated:
  - `<name>`: contains C++ source files to register ATen-compliant operators. Required by ExecuTorch runtime.
  - Input: most of the input fields are self-explainatory.
    - `deps`: kernel libraries - can be custom kernels or portable kernels (see portable kernel library [README.md](https://fburl.com/code/zlgs6zzf) on how to add more kernels) - needs to be provided. Selective build related targets should also be passed into the generated libraries through `deps`.
    - `define_static_targets`: if true we will generate a `<name>_static` library with static linkage. See docstring for more information.
    - `functions_yaml_target`: the target pointing to `functions.yaml`. See `ATen-compliant Operator Registration` section for more details.


We also provide selective build system to allow user to select operators from both `functions.yaml` and `custom_ops.yaml` into ExecuTorch build. See [Selective Build](https://www.internalfb.com/intern/staticdocs/executorch/docs/tutorials/custom_ops/#selective-build) section.



## Common Issues


### Missing operator

Example error message:
```
Nov 14 16:48:07 devvm11149.prn0.facebook.com bento[1985271]: [354870826409]Executor.cpp:260 Missing operator: [1] aten::squeeze_copy.dim_out

Nov 14 16:48:07 devvm11149.prn0.facebook.com bento[1985271]: [354870830000]Executor.cpp:267 In function init(), assert failed (num_missing_ops == 0): There are 1 operators missing from registration to Executor. See logs for details
```

This error message indicates that the operators are not registered into the ExecuTorch runtime.

For lean mode mode, please make sure the ATen-compliant operator schema is being added to your `functions.yaml`. For more guidance of how to write a `functions.yaml` file, please refer to [Declare the operator in a YAML file](https://www.internalfb.com/code/fbsource/xplat/executorch/kernels/portable/README.md).

For both ATen mode and lean mode, double check whether the operator is being selected correctly. One way to debug is to select all operator (follow the instruction in [et_operator_library](https://www.internalfb.com/intern/staticdocs/executorch/docs/tutorials/custom_ops/#apis)) and check if the error is gone.

### Undefined symbols

Example error message:
```
ld.lld: error: undefined symbol: torch::executor::native::add_out(torch::executor::Tensor const&, torch::executor::Tensor const&, torch::executor::Scalar const&, torch::executor::Tensor&)
>>> referenced by Functions.h:34 (buck-out/v2/gen/fbsource/9f23200ddcddc3cb/xplat/executorch/codegen/__generated_lib_combined__/out/Functions.h:34)
>>>               __objects__/UnboxingFunctionsEverything.cpp.o:(torch::executor::add_outf(torch::executor::Tensor const&, torch::executor::Tensor const&, torch::executor::Scalar const&, torch::executor::Tensor&)) in archive buck-out/v2/gen/fbsource/9f23200ddcddc3cb/xplat/executorch/codegen/__generated_lib__/libgenerated_lib.a
```
This is likely caused by one of the following:
* a typo in kernel definition that causes type mismatch.
* namespace caveats described in [Caveats](#caveats).
* other build isses such as dependency not being added to runtime target.
