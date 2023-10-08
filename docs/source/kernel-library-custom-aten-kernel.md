# Overview

At the last stage of [ExecuTorch model exporting](./export-overview.md), we lower the operators in the dialect to the _out variants_ of the [core ATen operators](./ir-ops-set-definition.md). Then we serialize these operator names into the model artifact. During runtime execution, for each operator name we will need to find the actual _kernels_, i.e., the C++ functions that do the heavy-lifting calculations and return results.

Portable kernel library is the in-house default kernel library, it’s easy to use and portable for most of the target backends. However it’s not optimized for performance, because it’s not specialized for any certain target. Therefore we provide kernel registration APIs for ExecuTorch users to easily register their own optimized kernels.


# Design Principles

**What do we support?** On the operator coverage side, the kernel registration APIs allow users to register kernels for all core ATen ops as well as custom ops, as long as the custom ops schemas are specified.

Notice that we also support _partial kernels, _for example the kernel only supports a subset of tensor dtypes and/or dim orders.

**Kernel contract**: kernels need to comply with the following requirements:

* Match the calling convention derived from operator schema. The kernel registration API will generate headers for the custom kernels as references.
* Satisfy the dtype constraints defined in edge dialect. For tensors with certain dtypes as arguments, the result of a custom kernel needs to match  the expected dtypes. The constraints are available in edge dialect ops.
* Gives correct result. We will provide a testing framework to automatically test the custom kernels.


# High Level Architecture

![](./_static/img/kernel-library-custom-aten-kernel.png)

ExecuTorch users are asked to provide:

1. the custom kernel library with C++ implementations

2. a yaml file associated with the library that describes what operators are being implemented by this library. For partial kernels, the yaml file also contains information on the dtypes and dim orders supported by the  kernel. More details in the API section.


## Workflow

At build time, the yaml files associated with kernel libraries will be passed to the _kernel resolver_ along with the model op info (see selective build doc) and the outcome is a mapping between a combination of operator names and tensor metadata, to kernel symbols. Then codegen tools will use this mapping to generate C++ bindings that connect the kernels to ExecuTorch runtime. ExecuTorch users need to link this generated library into their application to use these kernels.

At static object initialization time, kernels will be registered into the ExecuTorch kernel registry.

At runtime initialization stage, ExecuTorch will use the operator name and argument metadata as a key to lookup for the kernels. For example, with “aten::add.out” and inputs being float tensors with dim order (0, 1, 2, 3), ExecuTorch will go into the kernel registry and lookup for a kernel that matches the name and the input metadata.


# APIs

There are two sets of APIs: yaml files that describe kernel - operator mappings and codegen tools to consume these mappings.


## Yaml Entry for Core ATen Op Out Variant

Top level attributes:



* `op` (if the operator appears in `native_functions.yaml`) or `func` for custom operator. The value for this key needs to be the full operator name (including overload name) for `op` key, or a full operator schema (namespace, operator name, operator overload name and schema string), if we are describing a custom operator. For schema syntax please refer to this [instruction](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md).
* `kernels`: defines kernel information. It consists of `arg_meta` and `kernel_name`, which are bound together to describe "for input tensors with these metadata, use this kernel".
* `type_alias`(optional): we are giving aliases to possible dtype options. `T0: [Double, Float]` means `T0` can be one of `Double` or `Float`.
* `dim_order_alias`(optional): similar to `type_alias`, we are giving names to possible dim order options.

Attributes under `kernels`:



* `arg_meta`: a list of "tensor arg name" entries. The values for these keys are dtypes and dim orders aliases, that are implemented by the corresponding `kernel_name`. This being `null` means the kernel will be used for all types of input.
* `kernel_name`: the expected name of the C++ function that will implement this operator. You can put whatever you want to here, but you should follow the convention of replacing the `.` in the overload name with an underscore, and lowercasing all characters. In this example, `add.out` uses the C++ function named `add_out`. `add.Scalar_out` would become `add_scalar_out`, with a lowercase `S`. We support namespace for kernels, but note that we will be inserting a `native::` to the last level of namespace. So `custom::add_out` in the `kernel_name` will point to `custom::native::add_out`.

Some examples of operator entry:
```yaml
- op: add.out
  kernels:
    - arg_meta: null
      kernel_name: torch::executor::add_out
```
An out variant of a core ATen operator with a default kernel

ATen operator with a dtype/dim order specialized kernel (works for `Double` dtype and dim order needs to be (0, 1, 2, 3))
```yaml
- op: add.out
  type_alias:
    T0: [Double]
  dim_order_alias:
    D0: [[0, 1, 2, 3]]
  kernels:
    - arg_meta:
        self: [T0, D0]
        other: [T0 , D0]
        out: [T0, D0]
      kernel_name: torch::executor::add_out

```

## Custom Ops Yaml Entry

For custom ops (the ones that are not part of the out variants of core ATen opset) we need to specify the operator schema as well as a `kernel` section. So instead of `op` we use `func` with the operator schema. As an example, here’s a yaml entry for a custom op:
```yaml
- func: allclose.out(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False, bool dummy_param=False, *, Tensor(a!) out) -> Tensor(a!)
  kernels:
    - arg_meta: null
      kernel_name: torch::executor::allclose_out
```
The `kernel` section is the same as the one defined in core ATen ops. For operator schema, we are reusing the DSL defined in this [README.md](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md), with a few differences:


### Out variants only

ExecuTorch only supports out-style operators, where:


* The caller provides the output Tensor or Tensor list in the final position with the name `out`.
* The C++ function modifies and returns the same `out` argument.
    * If the return type in the YAML file is `()` (which maps to void), the C++ function should still modify `out` but does not need to return anything.
* The `out` argument must be keyword-only, which means it needs to follow an argument named `*` like in the `add.out` example below.
* Conventionally, these out operators are named using the pattern `<name>.out` or `<name>.<overload>_out`.

Since all output values are returned via an `out` parameter, ExecuTorch ignores the actual C++ function return value. But, to be consistent, functions should always return `out` when the return type is non-`void`.


### Can only return `Tensor` or `()`

ExecuTorch only supports operators that return a single `Tensor`, or the unit type `()` (which maps to `void`). It does not support returning any other types, including lists, optionals, tuples, or scalars like `bool`.


### Supported argument types

ExecuTorch does not support all of the argument types that core PyTorch supports. Here's a list of the argument types we currently support:
* Tensor
* int
* bool
* float
* str
* Scalar
* ScalarType
* MemoryFormat
* Device
* Optional<Type>
* List<Type>
* List<Optional<Type>>
* Optional<List<Type>>


## Build Tool Macros

We provide build time macros to help users to build their kernel registration library. The macro takes the yaml file describing the kernel library as well as model operator metadata, and packages the generated C++ bindings into a C++ library. The macro is available on both CMake and Buck2.


### CMake

`generate_bindings_for_kernels(functions_yaml, custom_ops_yaml)` takes a yaml file for core ATen op out variants and also a yaml file for custom ops, generate C++ bindings for kernel registration. It also depends on the selective build artifact generated by `gen_selected_ops()`, see selective build doc for more information. Then `gen_operators_lib` will package those bindings to be a C++ library. As an example:
```cmake
# SELECT_OPS_LIST: aten::add.out,aten::mm.out
gen_selected_ops("" "${SELECT_OPS_LIST}" "")

# Look for functions.yaml associated with portable libs and generate C++ bindings
generate_bindings_for_kernels(${EXECUTORCH_ROOT}/kernels/portable/functions.yaml "")

# Prepare a C++ library called "generated_lib" with _kernel_lib being the portable library, executorch is a dependency of it.
gen_operators_lib("generated_lib" ${_kernel_lib} executorch)

# Link "generated_lib" into the application:
target_link_libraries(executorch_binary generated_lib)

```

### Buck2

`executorch_generated_lib` is the macro that takes the yaml files and depends on the selective build macro `et_operator_library`. For an example:
```python
# Yaml file for kernel library
export_file(
  name = "functions.yaml"
)

# Kernel library
cxx_library(
  name = "add_kernel",
  srcs = ["add.cpp"],
)

# Selective build artifact, it allows all operators to be registered
et_operator_library(
  name = "all_ops",
  include_all_ops = True, # Select all ops in functions.yaml
)

# Prepare a generated_lib
executorch_generated_lib(
  name = "generated_lib",
  functions_yaml_target = ":functions.yaml",
  deps = [
    ":all_ops",
    ":add_kernel",
  ],
)

# Link generated_lib to ExecuTorch binary
cxx_binary(
 name = "executorch_bin",
 deps = [
  ":generated_lib",
 ],
)

```
