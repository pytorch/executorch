# Custom Operator Registration


## Introduction

Custom operator is defined in contrast with ATen-compliant operator.
ATen-compliant operators are those defined in
[`native_functions.yaml`](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml),
with their native functions (or kernels, we use these two terms interchangeably)
defined in ATen library. Custom operator lives out of ATen library and is most
likely associated with target model and hardware platform.

There are two types of usages of custom ops. The first type introduces custom ops into PyTorch model eager mode, whilst the second type only introduces custom ops during graph transformation.

## Step by step guide
There are two branches for this use case:
* ATen mode. In this case we expect the exported model to be able to run with ATen kernels (as well as custom kernels for custom ops).
* Lean mode. This requires custom op implementations using `ETensor`.

This also gets entangled with out variants of these custom ops. Generally, a custom op will require an out variant (with C++ implementation using `ETensor`) to be able to run on Executorch runtime.

In a nutshell, we need the following steps in order for a custom op to work on Executorch:
1. Register the custom op definition into PyTorch runtime so that they are visible to Executorch compiler.
2. Implement and register the implementation of it to PyTorch runtime. Do one of the following:
    1. Implement in Python and use [`library.py`](https://www.internalfb.com/code/fbsource/fbcode/caffe2/torch/library.py) API to register it
    2. Implement in C++, use `at::Tensor` or `exec_aten::Tensor` and use [`library.h`](https://www.internalfb.com/code/fbsource/fbcode/caffe2/torch/library.h) API to register it
3. Define an out variant of the custom op, implement it in C++ using `ETensor`. This step should also be trivial if we used `exec_aten::Tensor` in step 2.ii, since we can share the same logic for the two variants.
4. Create `custom_ops.yaml` for this operator, both functional and out variant, specify function schema and corresponding kernels. (See Common APIs for more info).
    1. In ATen mode, the C++ kernel implementation using `at::Tensor` will be linked.
    2. In lean mode, the C++ kernel implementation using `ETensor` will be linked.

### Case Study
Let's say a model uses a custom operator called `my_op::foo`. It's very common that we register them in eager mode like the following:
```c
// foo.cpp
at::Tensor foo(const at::Tensor& a) {
  at::Tensor result = a;
  // do something
  return result;
}

TORCH_LIBRARY(my_op, m) {
  m.def("foo", foo);
}

TORCH_LIBRARY_IMPL(my_op, CPU, m) {
  m.impl("foo), TORCH_FN(foo));
}
```

Alternatively this operator can be registered through Python API:
```python
# foo.py
from torch.library import impl, Library
lib = Library("my_op", "DEF")

lib.define("foo(Tensor a) -> Tensor")

@impl(lib, "foo", "CPU")
def foo(a):
  # do something
  return a
```
If we already have similar code checked in, that satisfies our steps 1 and 2 described above. If we start to implement this custom operator from scratch, we can implement it in either Python or C++, as long as we register it into the dispatcher.

Now if we want to use `my_op::foo` in Executorch, first we need to define a out variant of it and implement it in C++. Notice that we should implement it using `exec_aten` namespace so that it can be treated as either an `at::tensor` or an `ETensor` under the hood:
```c
// foo.cpp
namespace my_op {
namespace native {
exec_aten::Tensor& foo_out(const exec_aten::Tensor& a, exec_aten::Tensor& out) {
  // do something
  return out;
}
} // namespace native
} // namespace my_op
```
Along with a `custom_ops.yaml` that binds the function schema with the C++ implementation:
```yaml
- func: my_op::foo.out(Tensor a, *, Tensor(a!) out) -> Tensor(a!)
  variatns: function
  dispatch:
    CPU: my_op::foo_out # pointing to my_op::native::foo_out
```
We can generate code similar to the functional variant to register this out variant into PyTorch runtime. We can also generate code to register it into Executorch runtime.

For ATen mode and lean mode we can have two sets of generated libs:
```python
export_file(
  name = "custom_ops.yaml"
)

cxx_library(
  name = "foo",
  srcs = ["foo.cpp"],
)

executorch_generated_lib(
  name = "my_op_aten",
  custom_ops_yaml_target = ":custom_ops.yaml",
  custom_ops_aten_kernel_deps = [
    ":foo", # This means foo_out can be registered into PyTorch dispatcher as well.
  ],
  aten_mode = True,
  deps = [
    ":foo", # Kernel for Executorch runtime
  ],
)

executorch_generated_lib(
  name = "my_op_lean",
  custom_ops_yaml_target = ":custom_ops.yaml",
  custom_ops_aten_kernel_deps = [
    ":foo", # This means foo_out can be registered into PyTorch dispatcher as well.
  ],
  aten_mode = False,
  deps = [
    ":foo",
  ],
)
```
### Usage of generated lib
Here's a breakdown on what libraries we will generate and how to use them.
* `my_op_aten`:
    * `custom_ops_my_op_aten`: C++ library responsible to register `my_op::foo.out` into PyTorch runtime.
    * `my_op_aten`: C++ library responsible to register `my_op::foo.out` into Executorch runtime.
* `my_op_lean`:
    * `custom_ops_my_op_lean`: C++ library responsible to register `my_op::foo.out` into PyTorch runtime
    * `my_op_lean`: C++ library responsible to register `my_op::foo.out` into Executorch runtime.

So in the compiler we can register `my_op::foo.out` by loading a shared library:
```python
# custom_passes.py
torch.ops.load_library("//path/to:custom_ops_my_op_aten")

# we have access to my_op::foo.out
print(torch.ops.my_op.foo.out)
```
If we want to use this in Python unit test, we can add the library as a `preload_deps`:
```python
python_unittest(
    name = "test",
    srcs = ["test.py"],
    # Only use preload_deps to load "custom_ops_my_op_aten" in python_unittest
    # instead of in RnntModel.py. See D40133925 for more detail.
    preload_deps = [
        "//path/to:custom_ops_my_op_aten",
    ],
    deps = [
        ...
    ],
)
```
## Common APIs

To facilitate custom operator registration, we provide the following APIs:

- `custom_ops.yaml`: operator schema and kernel metadata are defined in this file, following the same syntax as in `native_functions.yaml`.
- `executorch_generated_lib`: the Buck rule to call Executorch codegen system and encapsulate generated C++ source files into libraries. Two libraries will be generated:
  - `<name>`: contains C++ source files to register both ATen-compliant operators and custom operators into Executorch runtime. Required by Executorch runtime.
  - `custom_ops_<name>`: contains C++ source files to register custom operators into PyTorch runtime. This library will be used by compiler but not necessarily Executorch runtime.
  - Input: most of the input fields are self-explainatory.
    - `deps`: kernel libraries - can be custom kernels or portable kernels (see portable kernel library [README.md](https://fburl.com/code/zlgs6zzf) on how to add more kernels) - needs to be provided. Selective build related targets can also be passed into the generated libraries through `deps`.
    - `define_static_targets`: if true we will generate a `<name>_static` library with static linkage. See docstring for more information.
    - `functions_yaml_target`: the target pointing to `functions.yaml`. See `ATen-compliant Operator Registration` section for more details.
    - `custom_ops_target`: the target pointing to `custom_ops.yaml`. Since custom operators are tightly coupled with Executorch users, each user will need to maintain their own `custom_ops.yaml` file.


We also provide selective build system to allow user to select operators from both `functions.yaml` and `custom_ops.yaml` into Executorch build. See [Selective Build](https://www.internalfb.com/intern/staticdocs/executorch/docs/tutorials/selective_build/) tutorial.

## Best Practices

- Out variant vs. functional variant: currently functionalization is not supported on custom mutating operators (e.g., out variants), hence the recommended way is to register both functional and out variants into `custom_ops.yaml`. During model authoring stage, the compiler recognize both variants and will perform a transform to replace functional variant with its corresponding out variant.
- Custom namespace support: we support custom namespace for both operator level and kernel level. It's recommended to use a different namespace than `aten` for custom operators, the definition should also live in a different namespace than `at::native`. See [README.md](https://fburl.com/code/kn4zexqm) for more information on namespaces.
- ATen library missing out variant: a lot of the need for custom operator is due to missing out variant in ATen library. If that happens, custom operator registration can be used to register those missing out variants as a short-term workaround. Please contact PyTorch Edge Team Portibility pillar to discuss a long-term approach. You can post in this [PyTorch Edge Users group](https://fb.workplace.com/groups/pytorch.edge.users) or reach out to [Executorch oncall](https://www.internalfb.com/omh/view/executorch/oncall_profile).

## Common Issues

### Missing out variants

Example error message:
```
[2022-11-14T11:57:40.588-08:00] ======================================================================
[2022-11-14T11:57:40.588-08:00] ERROR: test_end_to_end_executorch_dynamo (executorch.test.end2end.test_end2end_real_model_dynamo.ExecutorchDynamoTests)
[2022-11-14T11:57:40.588-08:00] ----------------------------------------------------------------------
[2022-11-14T11:57:40.588-08:00] Traceback (most recent call last):
[2022-11-14T11:57:40.588-08:00]   File "/data/sandcastle/boxes/eden-trunk-hg-fbcode-fbsource/buck-out/v2/gen/fbcode/9f23200ddcddc3cb/executorch/test/end2end/__test_end2end_real_model_dynamo__/test_end2end_real_model_dynamo#link-tree/torch/fx/passes/infra/pass_manager.py", line 271, in __call__
[2022-11-14T11:57:40.588-08:00]     res = fn(module)
[2022-11-14T11:57:40.588-08:00]   File "/data/sandcastle/boxes/eden-trunk-hg-fbcode-fbsource/buck-out/v2/gen/fbcode/9f23200ddcddc3cb/executorch/test/end2end/__test_end2end_real_model_dynamo__/test_end2end_real_model_dynamo#link-tree/torch/fx/passes/infra/pass_manager.py", line 35, in wrapped_fn
[2022-11-14T11:57:40.588-08:00]     res = fn(gm)
[2022-11-14T11:57:40.588-08:00]   File "/data/sandcastle/boxes/eden-trunk-hg-fbcode-fbsource/buck-out/v2/gen/fbcode/9f23200ddcddc3cb/executorch/test/end2end/__test_end2end_real_model_dynamo__/test_end2end_real_model_dynamo#link-tree/torch/fx/passes/infra/pass_base.py", line 44, in __call__
[2022-11-14T11:57:40.588-08:00]     self.ensures(graph_module)
[2022-11-14T11:57:40.588-08:00]   File "/data/sandcastle/boxes/eden-trunk-hg-fbcode-fbsource/buck-out/v2/gen/fbcode/9f23200ddcddc3cb/executorch/test/end2end/__test_end2end_real_model_dynamo__/test_end2end_real_model_dynamo#link-tree/executorch/exir/passes/__init__.py", line 212, in ensures
[2022-11-14T11:57:40.588-08:00]     raise RuntimeError(f"Missing out variants: {self.missing_out_vars}")
[2022-11-14T11:57:40.588-08:00] RuntimeError: Missing out variants: {'jarvis_nn_ops::attention_mask'}
```
This is likely caused by the out variant operator not defined or not linked to the binary/unittest target, so that it's missing from PyTorch runtime. Please follow the [Step by step guide](#step-by-step-guide) step 3 & 4 and [Case Study](#case-study), add the generated lib in the dependency and load the shared library like explained in [Usage of generated lib](#usage-of-generated-lib).

If the operator is still missing, double check whether the operator is being selected correctly. One way to debug is to select all operator (follow the instruction in [et_operator_library](#et_operator_library)) and check if the error is gone.

### Missing operator

Example error message:
```
Nov 14 16:48:07 devvm11149.prn0.facebook.com bento[1985271]: [354870826409]Executor.cpp:260 Missing operator: [1] aten::squeeze_copy.dim_out\n

Nov 14 16:48:07 devvm11149.prn0.facebook.com bento[1985271]: [354870830000]Executor.cpp:267 In function init(), assert failed (num_missing_ops == 0): There are 1 operators missing from registration to Executor. See logs for details\n
```

This error message indicates that the operators are not registered into the Executorch runtime. It's important to identify whether the missing operator is from ATen library or is a custom op. If the missing operator name starts with `aten` then it's an ATen operator, otherwise it's a custom op.

#### Missing ATen operator

Please make sure the ATen operator schema is being added to your `functions.yaml`. For more guidance of how to write a `functions.yaml` file, please refer to [Declare the operator in a YAML file](https://www.internalfb.com/code/fbsource/xplat/executorch/kernels/portable/README.md).

#### Missing custom operator

This means the custom operator is not being registered to Executorch runtime. A custom operator (functional variant) needs to be registered into PyTorch runtime so that it's available in compiler; a custom operator (out variant) needs to be registered into both PyTorch runtime and Executorch runtime, so that it can be  lowered and executed.

To fix this,
1. make sure the operator is being added to `custom_ops.yaml`, refer to [Declare the operator in a YAML file](https://www.internalfb.com/code/fbsource/xplat/executorch/kernels/portable/README.md) for details.
2. double check whether the operator is being selected correctly. One way to debug is to select all operator (follow the instruction in [et_operator_library](#et_operator_library)) and check if the error is gone.
