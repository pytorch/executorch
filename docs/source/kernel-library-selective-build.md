# Kernel Library Selective Build

_Selective build_ is a build mode on ExecuTorch that uses model metadata to guide ExecuTorch build. This build mode contains build tool APIs available on CMake. ExecuTorch users can use selective build APIs to build an ExecuTorch runtime binary with minimal binary size by only including operators required by models.

This document aims to help ExecuTorch users better use selective build, by listing out available APIs, providing an overview of high level architecture and showcasing examples.

Preread: [Overview of the ExecuTorch runtime](runtime-overview.md), [High-level architecture and components of ExecuTorch](getting-started-architecture.md)


## Design Principles

**Why selective build?** Many ExecuTorch use cases are constrained by binary size. Selective build can reduce the binary size of the ExecuTorch runtime without compromising support for a target model.

**What are we selecting?** Our core ExecuTorch library is around 50kB with no operators/kernels or delegates. If we link in kernel libraries such as the ExecuTorch in-house portable kernel library, the binary size of the whole application surges, due to unused kernels being registered into the ExecuTorch runtime. Selective build is able to apply a filter on the kernel libraries, so that only the kernels actually being used are linked, thus reducing the binary size of the application.

**How do we select?** Selective build provides APIs to allow users to pass in _op info_, operator metadata derived from target models. Selective build tools will gather these op info and build a filter for all kernel libraries being linked in.


## High Level Architecture



![](_static/img/kernel-library-selective-build.png)


Note that all of the selective build tools are running at build-time (to be distinguished from compile-time or runtime). Therefore selective build tools only have access to static data from user input or models.

The basic flow looks like this:



1. For each of the models we plan to run, we extract op info from it, either manually or via a Python tool. Op info will be written into yaml files and generated at build time.
2. An _op info aggregator _will collect these model op info and merge them into a single op info yaml file.
3. A _kernel resolver _takes in the linked kernel libraries as well as the merged op info yaml file, then makes a decision on which kernels to be registered into ExecuTorch runtime.


## Selective Build CMake Options

To enable selective build when building the executorch kernel libraries as part of a CMake build, the following CMake options are exposed. These options affect the `executorch_kernels` CMake target. Make sure to link this target when using selective build.

 * `EXECUTORCH_SELECT_OPS_YAML`: A path to a YAML file specifying the operators to include.
 * `EXECUTORCH_SELECT_OPS_LIST`: A string containing the operators to include.
 * `EXECUTORCH_SELECT_OPS_MODEL`: A path to a PTE file. Only operators used in this model will be included.
 * `EXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD`: If enabled, operators will be further specialized to only operator on the data types specified in the operator selection.

Note that `EXECUTORCH_SELECT_OPS_YAML`, `EXECUTORCH_SELECT_OPS_LIST`, and `EXECUTORCH_SELECT_OPS_MODEL` are mutually exclusive. Only one operator specifier directive is allowed.

As an example, to build with only operators used in mv2_xnnpack_fp32.pte, the CMake build can be configured as follows.
```
cmake .. -DEXECUTORCH_SELECT_OPS_MODEL=mv2_xnnpack_fp32.pte
```

## APIs

For fine-grained control, we expose a CMake macro [gen_selected_ops](https://github.com/pytorch/executorch/blob/main/tools/cmake/Codegen.cmake#L12) to allow users to specify op info:

```
gen_selected_ops(
  LIB_NAME              # the name of the selective build operator library to be generated
  OPS_SCHEMA_YAML       # path to a yaml file containing operators to be selected
  ROOT_OPS              # comma separated operator names to be selected
  INCLUDE_ALL_OPS       # boolean flag to include all operators
  OPS_FROM_MODEL        # path to a pte file of model to select operators from
  DTYPE_SELECTIVE_BUILD # boolean flag to enable dtye selection
)
```

The macro makes a call to gen_oplist.py, which requires a [distinct selection](https://github.com/BujSet/executorch/blob/main/codegen/tools/gen_oplist.py#L222-L228) of API choice. `OPS_SCHEMA_YAML`, `ROOT_OPS`, `INCLUDE_ALL_OPS`, and `OPS_FROM_MODEL` are mutually exclusive options, and should not be used in conjunction. 

### Select all ops

If this input is set to true, it means we are registering all the kernels from all the kernel libraries linked into the application. If set to true it is effectively turning off selective build mode.


### Select ops from schema yaml

Context: each kernel library is designed to have a yaml file associated with it. For more information on this yaml file, see [Kernel Library Overview](kernel-library-overview.md). This API allows users to pass in the schema yaml for a kernel library directly, effectively allowlisting all kernels in the library to be registered.


### Select root ops from operator list

This API lets users pass in a list of operator names. Note that this API can be combined with the API above and we will create a allowlist from the union of both API inputs.

### Select ops from model

This API lets users pass in a pte file of an exported model. When used, the pte file will be parsed to generate a yaml file that enumerates the operators and dtypes used in the model. 

### Dtype Selective Build

Beyond pruning the binary to remove unused operators, the binary size can further reduced by removing unused dtypes. For example, if your model only uses floats for the `add` operator, then including variants of the `add` operators for `doubles` and `ints` is unnecessary. The flag `DTYPE_SELECTIVE_BUILD` can be set to `ON` to support this additional optimization. Currently, dtype selective build is only supported with the model API described above. Once enabled, a header file that specifies only the operators and dtypes used by the model is created and linked against a rebuild of the `portable_kernels` lib. This feature is only supported for the portable kernels library; it's not supported for optimized, quantized or custom kernel libraries.

## Example Walkthrough

In [examples/selective_build/CMakeLists.txt](https://github.com/BujSet/executorch/blob/main/examples/selective_build/CMakeLists.txt#L48-L72), we have the following cmake config options:

1. `EXECUTORCH_SELECT_OPS_YAML`
2. `EXECUTORCH_SELECT_OPS_LIST`
3. `EXECUTORCH_SELECT_ALL_OPS`
4. `EXECUTORCH_SELECT_OPS_FROM_MODEL`
5. `EXECUTORCH_DTYPE_SELECTIVE_BUILD`

These options allow a user to tailor the cmake build process to utilize the different APIs, and results in different invocations on the `gen_selected_ops` [function](https://github.com/BujSet/executorch/blob/main/examples/selective_build/CMakeLists.txt#L110-L123). The following table describes some examples of how the invocation changes when these configs are set:

| Example cmake Call | Resultant `gen_selected_ops` Invocation |
| :----: | :---:| 
|<code><br>  cmake -D… -DSELECT_OPS_LIST="aten::add.out,aten::mm.out" <br></code> | <code><br>  gen_selected_ops("" "${SELECT_OPS_LIST}" "" "" "") <br></code> |
|<code><br> cmake -D… -DSELECT_OPS_YAML=ON <br></code> | <code><br>  set(_custom_ops_yaml ${EXECUTORCH_ROOT}/examples/portable/custom_ops/custom_ops.yaml) <br> gen_selected_ops("${_custom_ops_yaml}" "" "") <br></code> |
|<code><br> cmake -D… -DEXECUTORCH_SELECT_OPS_FROM_MODEL="model.pte.out" <br></code> | <code><br> gen_selected_ops("" "" "" "${_model_path}" "") <br></code> |
|<code><br> cmake -D… -DEXECUTORCH_SELECT_OPS_FROM_MODEL="model.pte.out" -DEXECUTORCH_DTYPE_SELECTIVE_BUILD=ON<br></code> | <code><br> gen_selected_ops("" "" "" "${_model_path}" "ON") <br></code> |
