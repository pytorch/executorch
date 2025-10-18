# Selective Build Examples
To optimize binary size of ExecuTorch runtime, selective build can be used. This folder contains examples to select only the operators needed for ExecuTorch build.

These examples showcase two flows - the simple way, using CMake options to configure the framework build, and an advanced flow - showcasing user-defined kernel targets including custom operators.

Prerequisite: finish the [setting up wiki](https://pytorch.org/executorch/main/getting-started-setup).

## Example 1 - Basic Flow

This example showcases using CMake options to control which operators are included. This approach should be preferred when not using
custom operators or additional kernel libraries beyond the standard kernels provided by ExecuTorch.

The code under the basic/ directory builds a minimal model runner binary which links to a selective kernel target. To build the
example with operators needed for the MobileNetV2 model, run the following commands:
```
# From the executorch directory
python -m examples.portable.scripts.export --model_name="mv2" # Create a PTE file for MobileNetV2
cd examples/selective_build/basic
mkdir cmake-out && cd cmake-out
cmake .. -DEXECUTORCH_SELECT_OPS_MODEL="../../mv2.pte" # Build with kernels needed for mv2.pte
cmake --build . -j8
./selective_build_test --model_path="../../mv2.pte" # Run the model with the selective kernel library
```

### CMake Options

The example commands above show use of the EXECUTORCH_SELECT_OPS_MODEL option to select operators used in a PTE file, but there are
several ways to provide the operator list. The options can be passed to CMake in the same way (during configuration) and are mutually
exclusive, meaning that only one of these options should be chosen.

 * `EXECUTORCH_SELECT_OPS_MODEL`: Select operators used in a .PTE file. Takes a path to the file.
 * `EXECUTORCH_SELECT_OPS_YAML`: Provide a list of operators from a .yml file, typically generated with the `codegen/tools/gen_oplist.py` script. See this script for usage information.
 * `EXECUTORCH_SELECT_OPS_LIST`: Provide a comma-separated list of operators to include. An example is included below.

Example operator list specification (passed as a CLI arg to the CMake configure command):
```
-DEXECUTORCH_SELECT_OPS_LIST="aten::convolution.out,\
aten::_native_batch_norm_legit_no_training.out,aten::hardtanh.out,aten::add.out,\
aten::mean.out,aten::view_copy.out,aten::permute_copy.out,aten::addmm.out,\
aten,aten::clone.out"
```

#### DType-Selective Build

To further reduce binary size, ExecuTorch can specialize the individual operators for only the dtypes (data types) used. For example, if
the model only calls add with 32-bit floating point tensors, it can drop parts of the code that handle integer tensors or other floating point types. This option is controlled by passing `-DEXECUTORCH_DTYPE_SELECTIVE_BUILD=ON` to CMake. It is only supported in conjunction
with the `EXECUTORCH_SELECT_OPS_MODEL` option and is not yet supported for other modes. It is recommended to enable this option when using `EXECUTORCH_SELECT_OPS_MODEL` as it provides significant size savings on top of the kernel selective build.

### How it Works

The CMake options described above are read by ExecuTorch framework build, which is referenced via `add_subdirectory` in basic/CMakeLists.txt. These options reflect in the `executorch_kernels` CMake target, which is linked against the example binary.

```cmake
# basic/CMakeLists.txt
target_link_libraries(
  selective_build_test
  PRIVATE executorch_core extension_evalue_util extension_runner_util
          gflags::gflags executorch_kernels
)
```

To use selective build in a user CMake project, take the following steps:
 * Reference the executorch framework via `add_subdirectory`.
 * Add `executorch_kernels` as a dependency (via target_link_libraries).
 * Set CMake options at build time or in user CMake code.

To use the CMake-build framework libraries from outside of the CMake ecosystem, link against libexecutorch_selected_kernels.

## Example 2 - Advanced Flow for Custom Ops and Kernel Libraries

This example showcases defined a custom kernel target. This option can be used when defining custom operators or integrating with
kernel libraries not part of the standard ExecuTorch build.

The code under the advanced/ directory builds a minimal model runner binary which links to a user-defined kernel library target. To run a model with a simple custom operator, run the following commands:
```
# From the executorch directory
python -m examples.portable.custom_ops.custom_ops_1 # Create a model PTE file
cd examples/selective_build/basic
mkdir cmake-out && cd cmake-out
cmake .. -DEXECUTORCH_SELECT_OPS_MODEL="../../custom_ops_1.pte" -DEXECUTORCH_EXAMPLE_USE_CUSTOM_OPS=ON # Build with kernels needed for the model
cmake --build . -j8
./selective_build_test --model_path="../../custom_ops_1.pte" # Run the model with the selective kernel library
```

### CMake Options

The CMake logic in `advanced/CMakeLists.txt` respects the CMake options described in the basic flow, as well as the following options:

 * `EXECUTORCH_EXAMPLE_USE_CUSTOM_OPS`: Build and link some simple custom operators.
 * `EXECUTORCH_EXAMPLE_SELECT_ALL_OPS`: Build a kernel target with all available operators.

### How it Works

The build logic in `advanced/CMakeLists.txt` uses the `gen_selected_ops`, `generate_bindings_for_kernels`, and `gen_operators_lib` CMake functions to define an operator target. See [Kernel Library Selective Build](https://docs.pytorch.org/executorch/main/kernel-library-selective-build.html) for more information on selective build.

```cmake
gen_selected_ops(
  LIB_NAME
  "select_build_lib"
  OPS_SCHEMA_YAML
  "${_custom_ops_yaml}"
  ROOT_OPS
  "${EXECUTORCH_SELECT_OPS_LIST}"
  INCLUDE_ALL_OPS
  "${EXECUTORCH_SELECT_ALL_OPS}"
  OPS_FROM_MODEL
  "${EXECUTORCH_SELECT_OPS_MODEL}"
  DTYPE_SELECTIVE_BUILD
  "${EXECUTORCH_DTYPE_SELECTIVE_BUILD}"
)

generate_bindings_for_kernels(
  LIB_NAME
  "select_build_lib"
  FUNCTIONS_YAML
  ${EXECUTORCH_ROOT}/kernels/portable/functions.yaml
  CUSTOM_OPS_YAML
  "${_custom_ops_yaml}"
  DTYPE_SELECTIVE_BUILD
  "${EXECUTORCH_DTYPE_SELECTIVE_BUILD}"
)

gen_operators_lib(
  LIB_NAME
  "select_build_lib"
  KERNEL_LIBS
  ${_kernel_lib}
  DEPS
  executorch_core
  DTYPE_SELECTIVE_BUILD
  "${EXECUTORCH_DTYPE_SELECTIVE_BUILD}"
)
```

To link against this target, the top-level binary target declares a dependency on `select_build_lib`, which is the library name defined by the above function invocations. To use outside of the CMake ecosystem, link against libselect_build_lib.

See `test_selective_build.sh` for additional build examples.
