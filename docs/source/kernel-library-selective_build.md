# Kernel Library Selective Build

_Selective build_ is a build mode on ExecuTorch that uses model metadata to guide ExecuTorch build. This build mode contains build tool APIs available on both CMake and buck2. ExecuTorch users can use selective build APIs to build an ExecuTorch runtime binary with minimal binary size by only including operators required by models.

This document aims to help ExecuTorch users better use selective build, by listing out available APIs, providing an overview of high level architecture and showcasing examples.

Preread: Overview of the ExecuTorch runtime, High-level architecture and components of ExecuTorch


## Design Principles

**Why selective build?** Many ExecuTorch use cases are constrained by binary size. Selective build can reduce the binary size of the ExecuTorch runtime without compromising support for a target model.

**What are we selecting?** Our core ExecuTorch library is around 50kB with no operators/kernels or delegates. If we link in kernel libraries such as the ExecuTorch in-house portable kernel library, the binary size of the whole application surges, due to unused kernels being registered into the ExecuTorch runtime. Selective build is able to apply a filter on the kernel libraries, so that only the kernels actually being used are linked, thus reducing the binary size of the application.

**How do we select? **Selective build provides APIs to allow users to pass in _op info_, operator metadata derived from target models. Selective build tools will gather these op info and build a filter for all kernel libraries being linked in.


## High Level Architecture



![](./_static/img/kernel-library-selective_build.png)


Note that all of the selective build tools are running at build-time (to be distinguished from compile-time or runtime). Therefore selective build tools only have access to static data from user input or models.

The basic flow looks like this:



1. For each of the models we plan to run, we extract op info from it, either manually or via a Python tool. Op info will be written into yaml files and generated at build time.
2. An _op info aggregator _will collect these model op info and merge them into a single op info yaml file.
3. A _kernel resolver _takes in the linked kernel libraries as well as the merged op info yaml file, then makes a decision on which kernels to be registered into ExecuTorch runtime.


## APIs

We expose build macros for CMake and Buck2, to allow users specifying op info.

On CMake:

[gen_selected_ops](https://github.com/pytorch/executorch/blob/main/build/Codegen.cmake#L12)

On Buck2:

[et_operator_library](https://github.com/pytorch/executorch/blob/main/shim/xplat/executorch/codegen/codegen.bzl#L44C21-L44C21)

Both of these build macros take the following inputs:


### Select all ops

If this input is set to true, it means we are registering all the kernels from all the kernel libraries linked into the application. If set to true it is effectively turning off selective build mode.


### Select ops from schema yaml

Context: each kernel library is designed to have a yaml file associated with it. For more information on this yaml file, see here (TODO: add link to kernel library documentation). This API allows users to pass in the schema yaml for a kernel library directly, effectively allowlisting all kernels in the library to be registered.


### Select root ops from operator list

This API lets users pass in a list of operator names. Note that this API can be combined with the API above and we will create a allowlist from the union of both API inputs.


### Select from model (WIP)

This API takes a model and extracts all op info from it.


## Example Walkthrough


### Buck2 example

Let’s take a look at the following build:

```
# Select a list of operators: defined in `ops`
et_operator_library(
    name = "select_ops_in_list",
    ops = [
        "aten::add.out",
        "aten::mm.out",
    ],
)
```
This target generates the yaml file containing op info for these two ops.

In addition to that, if we want to select all ops from a kernel library, we can do:

```
# Select all ops from a yaml file
et_operator_library(
    name = "select_ops_from_yaml",
    ops_schema_yaml_target = "//executorch/examples/portable/custom_ops:custom_ops.yaml",
)
```
Then in the kernel registration library we can do:
```
executorch_generated_lib(
    name = "select_ops_lib",
    custom_ops_yaml_target = "//executorch/examples/portable/custom_ops:custom_ops.yaml",
functions_yaml_target = "//executorch/kernels/portable:functions.yaml",
    deps = [
        "//executorch/examples/portable/custom_ops:custom_ops_1", # kernel library
        "//executorch/examples/portable/custom_ops:custom_ops_2", # kernel library
  "//executorch/kernels/portable:operators", # kernel library
        ":select_ops_from_yaml",
  ":select_ops_in_list",
    ],
)
```
Notice we are allowlisting both add.out, mm.out from the list, and the ones from the schema yaml (`custom_ops.yaml`).


### CMake example

In CMakeLists.txt we have the following logic:
```cmake
set(_kernel_lib)
if(SELECT_ALL_OPS)
  gen_selected_ops("" "" "${SELECT_ALL_OPS}")
elseif(SELECT_OPS_LIST)
  gen_selected_ops("" "${SELECT_OPS_LIST}" "")
elseif(SELECT_OPS_YAML)
 set(_custom_ops_yaml ${EXECUTORCH_ROOT}/examples/portable/custom_ops/custom_ops.yaml)
  gen_selected_ops("${_custom_ops_yaml}" "" "")
endif()
```
Then when calling CMake, we can do:

```
cmake -D… -DSELECT_OPS_LIST="aten::add.out,aten::mm.out”
```

Or

```
cmake -D… -DSELECT_OPS_YAML=ON
```

To select from either an operator name list or a schema yaml from kernel library.
