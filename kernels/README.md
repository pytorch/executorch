This subtree contains operator implementations that ExecuTorch clients can use and
contribute to. For internal users, please see `executorch/kernels/fb/README.md`.

## Layout

- `kernels`: Contains implementations and tests for the operators defined
  in the YAML files.
  - `kernels/portable/cpu`: Pure C++ implementations of the operators defined in the
    YAML files.
  - `kernels/optimized/cpu`: Optimized C++ implementations of the operators defined in the
    YAML files, for specific hardware platforms.
  - `kernels/aten`: A thin wrapper layer to hookup ATen library into ExecuTorch.
  - `kernels/test`: Tests for all operator implementations. Since all
    implementations should behave identically, the same tests should pass for
    all target types.

## Help & Improvements

If you have problems or questions, or have suggestions for ways to make
implementation and testing better, please contact [Dave
Bort](https://fb.workplace.com/profile.php?id=100042415022179), [Mengwei
Liu](https://fb.workplace.com/profile.php?id=100024007250862), or [Martin
 Yuan](https://fb.workplace.com/profile.php?id=100020734910364) on the PyTorch
Edge team.

## Contributing

Please follow these steps and guidelines when adding a new operator
implementation to this library. The goals of these guidelines are to:
- Make it straightforward to add new operator implementations.
- Ensure that the operator implementations are of high quality, and are easy to
  maintain.
- Make it easy for users to find available operator implementations, and to
  trust in their quality and behavioral stability.

### Your code must be compatible with ExecuTorch types

ExecuTorch does not use `at::Tensor`, `at::ScalarType`, `c10::Scalar`, or any of
the types defined by PyTorch core in the `at` or `c10` namespaces. To retain
tigher control over CPU and memory runtime behavior, ExecuTorch reimplements
compatible but restricted subsets of those types.

[`//runtime/core/exec_aten/exec_aten.h`](https://github.com/pytorch/executorch/blob/main/runtime/core/exec_aten/exec_aten.h)
contains the mapping between ATen/c10 types and the ExecuTorch types. The
ExecuTorch types are defined in other headers in that same directory,
[`//runtime/core/portable_type/`](https://github.com/pytorch/executorch/tree/main/runtime/core/portable_type).

The ExecuTorch types are source-compatible with the ATen/c10 types; if you write
code that works with the ExecuTorch types, then that same code should work when
built against ATen/c10. But, there are features of `at::Tensor` and other
ATen/c10 types that may not be present. In many cases this is intentional, but
in other cases we can consider adding the missing features.

### Declare the operator in a YAML file

We use yaml files to declare the ATen operators or custom operators being implemented by this kernel library.

Before implementing, the operator must be declared in exactly one of the
operator YAML files:
- [`//kernels/portable/functions.yaml`](https://github.com/pytorch/executorch/blob/main/kernels/portable/functions.yaml)
  - Add your entry here if your operator overload (e.g., `op: add.out`)
    appears in the core pytorch file
    [`pytorch/aten/src/ATen/native/native_functions.yaml`](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml).
  - Also add your entry to [`//kernels/aten/functions.yaml`](https://github.com/pytorch/executorch/blob/main/kernels/aten/functions.yaml) for test coverage.
- [`//kernels/portable/custom_ops.yaml`](https://github.com/pytorch/executorch/blob/main/kernels/portable/custom_ops.yaml)
  - Add your entry here if your operator overload does *not* appear in the core pytorch `native_functions.yaml`.

The next sections describe how to add a yaml entry.

#### YAML Schema

This YAML file schema is a DSL to decribe the operators and the kernels that implement them. This YAML file is a contract between AOT model export and runtime execution, that if followed correctly, can make sure ExecuTorch runtime be able to link the C++ implementation of an operator to the exported model artifact. Here are some rules of writing up your own YAML files.

**Out variants only**

ExecuTorch only supports out-style operators, where:
- The caller provides the output Tensor or Tensor list in the final position
  with the name `out`.
- The C++ function modifies and returns the same `out` argument.
  - If the return type in the YAML file is `()` (which maps to void), the C++
    function should still modify `out` but does not need to return anything.
- The `out` argument must be keyword-only, which means it needs to follow an
  argument named `*` like in the `add.out` example below.
- Conventionally, these out operators are named using the pattern `<name>.out`
  or `<name>.<overload>_out`.

Since all output values are returned via an `out` parameter, ExecuTorch ignores
the actual C++ function return value. But, to be consistent, functions should
always return `out` when the return type is non-`void`.

**Can only return `Tensor` or `()`**

ExecuTorch only supports operators that return a single `Tensor`, or the unit
type `()` (which maps to `void`). It does not support returning any other types,
including lists, optionals, tuples, or scalars like `bool`.

**Supported argument types**

ExecuTorch does not support all of the argument types that core PyTorch
supports. See [this
spreadsheet](https://docs.google.com/spreadsheets/d/1uArc0r1Yq1QSeyRJZKzZ8Wkz0eS9TsM39ghmMAZCXDA/edit#gid=0)
for the list of supported and unsupported types.
<!-- TODO(dbort): Once that list stablizes, move to a table in this file
so that external users can see it. -->

**Functions only, no methods**

ExecuTorch does not support Tensor methods, and assumes `variants: function` for
all operators. Entries like `variants: method` or `variants: function, method`
will be ignored.

#### Add your operator entry

Some examples of operator entry:

ATen operator with a default kernel
```
- op: add.out
  kernels:
    - arg_meta: null
      kernel_name: torch::executor::add_out
```

ATen operator with a dtype/dim order specialized kernel (works for `Double` dtype and dim order needs to be (0, 1, 2, 3))
```
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

Custom operator with a default kernel
```
- func: allclose.out(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False, bool dummy_param=False, *, Tensor(a!) out) -> Tensor(a!)
  kernels:
    - arg_meta: null
      kernel_name: torch::executor::allclose_out
```

Top level attributes:
* `op` (if the operator appears in `native_functions.yaml`) or `func` for custom operator. The value for this key needs to be the full operator name (including overload name) for `op` key, or a full operator schema (namespace, operator name, operator overload name and schema string). For schema syntax please refer to this [instruction](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md).

* `kernels`: this entry is used to define the information of kernels. It consists of `arg_meta` and `kernel_name`, they are bound together to describe "for input tensors with these metadata, use this kernel".
* `type_alias`(optional): we are giving aliases to possible dtype options. `T0: [Double, Float]` means `T0` can be one of `Double` or `Float`.
* `dim_order_alias`(optional): similar to `type_alias`, we are giving names to possible dim order options.

Attributes under `kernels`:
* `arg_meta`: a list of "tensor arg name" entries. The value for these keys are dtypes and dim orders alias, that are implemented by the corresponding `kernel_name`. This being `null` means the kernel will be used for all types of input.
* `kernel_name`: the expected name of the
C++ function that will implement this operator. You can put whatever you want to
here, but you should follow the convention of replacing the `.` in the overload
name with an underscore, and lowercasing all characters. In this example,
`add.out` uses the C++ function named `add_out`. `add.Scalar_out` would become `add_scalar_out`, with a lowercase `S`. We support namespace for kernels, but note that we will be inserting a `native::` to the last level of namespace. So `custom::add_out` in the `kernel_name` will point to `custom::native::add_out`.

### Find operator base name

The base name is the part of the operator name before the `.`, excluding any
trailing underscores. The rest of this document refer to this as `<name>`.

E.g., these operator overloads all have a base name of `add`:
- `add.Scalar`
- `add.Tensor`
- `add.out`
- `add_.Tensor`

So, if you were implementing `add.out` then your operator base name would be
`add`, and you would replace `<name>` with `add` everywhere below.

### Selective build

When using macros that require a `NAME` argument, eg. `#define ET_SWITCH_REAL_TYPES_AND(ADDITIONAL, TYPE, CONTEXT, NAME, CTYPE_ALIAS, ...)`, make sure to pass in the same operator name defined in `functions.yaml`. This is the base name + variant, eg. `add.out`, `add.Scalar_out`. The function name is required for dtype selective build, which matches against the operator names and dtypes present in a model.

### Overview of files and targets

For the operator base name `<name>`, you should work with these files. Sections below give more details about what they should contain.

- `./kernels/portable/cpu/op_<name>.cpp`: The implementations of operator overloads
  with base name `<name>`. This is the file that clients will link into their
  runtimes.
- `./kernels/portable/CMakeLists.txt`: The CMake build file for all the
  `op_<name>.cpp` files in the same directory.
- `./kernels/test/op_<name>_test.cpp`: Unit tests for the operator overloads
  with base name `<name>`.
  - Note that tests under this directory are for portable kernel specific. To
    share tests between multiple kernels, we can put tests in ../test.
  - Note that the tests do not live under `cpu`; tests should be
    implementation-agnostic. This will let us run the same tests against all
    implementations of a given operator, which should behave identically.
- `./kernels/test/CMakeLists.txt`: The CMake build file for all the
  `op_<name>_test.cpp` files in the same directory.

For an example, see the `add` operator (note that these are slightly different
from the `add` examples in this doc):
- [`executorch/kernels/portable/cpu/op_add.cpp`](https://github.com/pytorch/executorch/blob/main/kernels/portable/cpu/op_add.cpp):
  Implementations.
- [`./kernels/portable/CMakeLists.txt`](https://github.com/pytorch/executorch/blob/main/kernels/portable/CMakeLists.txt):
  Build portable ops.
- [`executorch/kernels/portable/test/op_add_test.cpp`](https://github.com/pytorch/executorch/blob/main/kernels/test/op_add_test.cpp):
  Unit tests.
- [`./kernels/test/CMakeLists.txt`](https://github.com/pytorch/executorch/blob/main/kernels/test/CMakeLists.txt):
  Build kernel tests.

### Add the operator implementation to CMakeLists.txt

The portable operator files are collected by [`./kernels/portable/CMakeLists.txt`](https://github.com/pytorch/executorch/blob/main/kernels/portable/CMakeLists.txt) with a glob on `./kernels/portable/cpu/*.cpp`. Ensure your operator file is in that directory.

NOTE: a given `op_<name>` cannot implement both ATen-compatible and
non-ATen-compatible (i.e., custom) operators. We suggest adding the suffix
`_custom` if necessary: e.g., `op_add` for ATen-compatible overloads of
the `add` operator, and `op_add_custom` for non-ATen-compatible overloads.

NOTE: An `op_<name>` may not have dependencies outside of `//executorch`.
This library is intended to be portable, open-sourceable, and self-contained.

### Create a skeleton .cpp file for the operator implementation

If not already present, create the file
`executorch/kernels/portable/cpu/op_<name>.cpp`, which should follow the
pattern:
```
// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

namespace {
  // <helper code>
} // namespace

// <operator overload implementations>

} // namespace native
} // namespace executor
} // namespace torch
```

### Find the function signature for the operator overload

When you add an entry to the YAML file, the codegen tools will generate an
expected function signature for you to implement in a file called
`NativeFunctions.h`. To build and find that generated header:

1. Build executorch
```
cmake -DCMAKE_INSTALL_PREFIX=cmake-out \
          -DCMAKE_BUILD_TYPE=Release \
          -DPYTHON_EXECUTABLE=python \
          -Bcmake-out .
cmake --build cmake-out -j9 --target install --config Release
```
2. The generated `NativeFunctions.h` file is located in
```
cmake-out/kernels/portable/portable_ops_lib/NativeFunctions.h
```

Since this header is generated from the YAML files, re-run the script if you have modified your
operator's entry in those files.

Open the file and look for the function with the same name that you earlier
added in the YAML file. For `add_out`, this might look like
```
TORCH_API torch::executor::Tensor & add_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out);
```

This is the function signature that you will need to implement.

### Add a stub implementation

Now that you have your function signature, add a stub to the `op_<name>.cpp`
file that just returns the `out` argument. For example:
```
Tensor& add_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  return out;
}
```

Note that you should drop the `TORCH_API` attribute, and should drop `at::`.

### Create a skeleton test .cpp file

If not already present, create the file
`executorch/kernels/portable/test/op_<name>_test.cpp`. Here's a suggested
starting point:
```
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::native::<operator_function_name>;
using torch::executor::testing::IsCloseTo;
using torch::executor::testing::TensorFactory;

TEST(Op<Name>Test, SmokeTest) {
  TensorFactory<ScalarType::Int> tf;

  Tensor a = tf.make(/*sizes=*/{2, 2}, /*data=*/{1, 1, 1, 1}):
  Tensor b = tf.ones(/*sizes=*/{2, 2}):
  Tensor z = tf.zeros(/*sizes=*/{2, 2}):

  EXPECT_EQ(a, b); // Exact equality
  EXPECT_THAT(a, IsCloseTo(b)); // For floating-point tensors

  EXPECT_NE(a, z);
  EXPECT_THAT(a, Not(IsCloseTo(z)));
}
```

### Add operator test to CMakeLists.txt

Now, we have to add this to [executorch/kernels/tests/CMakeLists.txt](https://github.com/pytorch/executorch/blob/main/kernels/test/CMakeLists.txt). Note that this builds all the kernel tests.

For portable kernels, add your test file to [`all_test_sources`](https://github.com/pytorch/executorch/blob/main/kernels/test/CMakeLists.txt#L69).

For optimized kernels, add your test file to [`_optimized_kernels_test_sources](https://github.com/pytorch/executorch/blob/main/kernels/test/CMakeLists.txt#L230).

### Implement and test the operator

You should now be able to implement and test your operator. It's helpful to see
how other operators do it, so take a look at `op_add`:
- [`executorch/kernels/portable/cpu/op_add.cpp`](https://github.com/pytorch/executorch/blob/main/kernels/portable/cpu/op_add.cpp)
- [`executorch/kernels/portable/test/op_add_test.cpp`](https://github.com/pytorch/executorch/blob/main/kernels/test/op_add_test.cpp):

Check out how it uses helper macros like `ET_CHECK_SAME_SHAPE_AND_DTYPE` and
`ET_FORALL_REAL_TYPES` when implementing the operator, and test helpers like
`TensorFactory` and `IsCloseTo()` when testing.

Once you have your operator and corresponding tests in place, we can try it out.

1. Build ExecuTorch.
```
cmake . \
  -DCMAKE_INSTALL_PREFIX=cmake-out \
  -DEXECUTORCH_USE_CPP_CODE_COVERAGE=ON \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_DEVTOOLS=ON \
  -DEXECUTORCH_BUILD_VULKAN=OFF \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -Bcmake-out

cmake --build cmake-out -j9 --target install
```
2. Build gtest.
```
mkdir -p third-party/googletest/build
cd third-party/googletest/build
cmake .. -DCMAKE_INSTALL_PREFIX=.
make -j4
make install
cd ../../../
```

3. Build kernel tests.
```
cmake kernels/test \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_INSTALL_PREFIX=cmake-out \
  -DEXECUTORCH_USE_CPP_CODE_COVERAGE=ON \
  -DCMAKE_PREFIX_PATH="$(pwd)/third-party/googletest/build" \
  -Bcmake-out/kernels/test
cmake --build cmake-out/kernels/test -j9
```
4. Run tests. You should see your test here.
```
./cmake-out/kernels/test/portable_kernels_test
./cmake-out/kernels/test/optimized_kernels_test
```

#### Implementation restrictions

To reduce dependencies and size, to ensure portability, and to conform to the
restrictions of embedded environments, your operator implementations:

- Must not include C++ stdlib headers, or use C++ stdlib types. For example,
  `string`/`basic_string`, `vector`, `unordered_map`, `cout`, `unique_pointer`
  must not be used.
- Must not dynamically allocate memory, or cause memory to be dynamically
  allocated. All non-stack memory must be provided as a function parameter by
  the caller, typically via an `out` parameter or another tensor parameter to be
  used as scratch space.
  - This includes direct calls to `new`, `malloc`, `realloc`, etc., as well as
    operations that allocate under the hood like `make_unique`, or the creation
    of `vector` or `string`, for example.
- Must be stateless.
- Must be thread-safe. Note that the ExecuTorch environment does not provide
  a locking construct, so this means that operator implementations must not
  modify global memory.
- Must work in an environment without threads. This, along with the stateless
  requirement, means that thread local storage must not be used.
- Must not use `stdout`, `stderr`, or other file/stream IO via `printf`/`cout`
  etc.; instead, use `ET_LOG` from `executorch/runtime/platform/log.h`.
- Must not use `assert()`. Instead use `ET_CHECK` and other macros from
  `executorch/runtime/platform/assert.h`.
- Must not raise exceptions. Instead use `ET_CHECK` and other macros from
  `executorch/runtime/platform/assert.h`.

Note that not all of these apply to *every* ExecuTorch-compatible operator
implementation, only those included in this portable library.

For example, a target-specfic custom operator that initiates a DMA copy would be
stateful, and would probaby modify global memory, but it would need to use
target-specific APIs to do so. But, since this library is only for portable
operator implementations, the operators it contains can't depend on
target-specific APIs like that.

### Shared kernel tests (executorch/kernels/test)
The portable kernel implementation and its corresponding tests can be used as a
reference for other kernels. We can also share the test cases in
`//executorch/kernels/test`, which contains common resources for kernel testing.

*generate_wrapper* generates a header FunctionHeaderWrapper.h, which simply
includes the corresponding Functions.h file for the specified kernel:
`#include <executorch/kernels/{}/Functions.h>`. With that, the test sources don't need to know
about which kernel we are testing and which Functions.h we should use.

With *_common_op_test* we use a single test source file (op_<op>_test.cpp) at this directory.
We automatically find the corresponding registered dispatch function through Funcitons.h, so
it can be used to test multiple kernels.

In <kernel>/test/ we can put kernel-specific test cases.

*supported_features* is used to distinguish between different kernel features. For example,
ATen supports mixing input and output dtype while portable doesn't. When we expect death in
portable testing in such case, we can check the supported features by the running kernel and
bypass if it's supported.
- The default value of supported features is in test/supported_features.yaml
- Each kernel needs to override its supported features in <kernel>/test/supported_features_def.yaml.
  See example in supported_features_def_example.yaml.
- This ensures that all kernels can share the same c++ test case source
