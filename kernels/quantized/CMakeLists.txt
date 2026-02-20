# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Kernel library for quantized operators. Please this file formatted by running:
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
cmake_minimum_required(VERSION 3.19)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

set(_common_compile_options -Wno-deprecated-declarations)

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)

# Quantized ops kernel sources TODO(larryliu0820): use buck2 to gather the
# sources
list(TRANSFORM _quantized_kernels__srcs PREPEND "${EXECUTORCH_ROOT}/")
# Generate C++ bindings to register kernels into both PyTorch (for AOT) and
# Executorch (for runtime). Here select all ops in quantized.yaml
set(_yaml_file ${CMAKE_CURRENT_LIST_DIR}/quantized.yaml)
gen_selected_ops(LIB_NAME "quantized_ops_lib" OPS_SCHEMA_YAML "${_yaml_file}")

# Expect gen_selected_ops output file to be selected_operators.yaml
generate_bindings_for_kernels(
  LIB_NAME "quantized_ops_lib" CUSTOM_OPS_YAML "${_yaml_file}"
)
message("Generated files ${gen_command_sources}")

# Not targeting Xcode, because the custom command generating
# kernels/quantized/selected_operators.yaml is attached to multiple targets:
# quantized_ops_aot_lib quantized_ops_lib but none of these is a common
# dependency of the other(s). This is not allowed by the Xcode "new build
# system".
if(NOT CMAKE_GENERATOR STREQUAL "Xcode"
   AND EXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT
)
  # Not targeting ARM_BAREMETAL as aot_lib depends on incompatible libraries
  if(NOT EXECUTORCH_BUILD_ARM_BAREMETAL)
    set(_quantized_aot_ops
        "quantized_decomposed::add.out"
        "quantized_decomposed::choose_qparams.Tensor_out"
        "quantized_decomposed::choose_qparams_per_token_asymmetric.out"
        "quantized_decomposed::dequantize_per_channel.out"
        "quantized_decomposed::dequantize_per_tensor.out"
        "quantized_decomposed::dequantize_per_tensor.Tensor_out"
        "quantized_decomposed::dequantize_per_token.out"
        "quantized_decomposed::mixed_linear.out"
        "quantized_decomposed::mixed_mm.out"
        "quantized_decomposed::quantize_per_channel.out"
        "quantized_decomposed::quantize_per_tensor.out"
        "quantized_decomposed::quantize_per_tensor.Tensor_out"
        "quantized_decomposed::quantize_per_token.out"
    )
    gen_selected_ops(
      LIB_NAME "quantized_ops_aot_lib" ROOT_OPS ${_quantized_aot_ops}
    )
    # Expect gen_selected_ops output file to be
    # quantized_ops_aot_lib/selected_operators.yaml
    generate_bindings_for_kernels(
      LIB_NAME "quantized_ops_aot_lib" CUSTOM_OPS_YAML "${_yaml_file}"
    )
    # Build a AOT library to register quantized ops into PyTorch. This is a
    # hack.
    set(_quantized_sources
        ${_quantized_kernels__srcs}
        ${EXECUTORCH_ROOT}/kernels/portable/cpu/util/reduce_util.cpp
        ${EXECUTORCH_ROOT}/runtime/core/exec_aten/util/tensor_util_aten.cpp
    )
    gen_custom_ops_aot_lib(
      LIB_NAME "quantized_ops_aot_lib" KERNEL_SOURCES "${_quantized_sources}"
    )

    # Register quantized ops to portable_lib, so that they're available via
    # pybindings.
    if(TARGET portable_lib)
      add_library(quantized_pybind_kernels_lib ${_quantized_kernels__srcs})
      target_link_libraries(
        quantized_pybind_kernels_lib PRIVATE portable_lib executorch_core
                                             kernels_util_all_deps
      )
      target_compile_options(
        quantized_pybind_kernels_lib PUBLIC ${_common_compile_options}
      )
      target_include_directories(
        quantized_pybind_kernels_lib PUBLIC "${_common_include_directories}"
      )
      gen_selected_ops(
        LIB_NAME "quantized_ops_pybind_lib" OPS_SCHEMA_YAML "${_yaml_file}"
      )
      generate_bindings_for_kernels(
        LIB_NAME "quantized_ops_pybind_lib" CUSTOM_OPS_YAML "${_yaml_file}"
      )
      # Build a library for pybind usage. quantized_ops_pybind_lib: Register
      # quantized ops kernels into Executorch runtime for pybind.
      gen_operators_lib(
        LIB_NAME "quantized_ops_pybind_lib" KERNEL_LIBS
        quantized_pybind_kernels_lib DEPS portable_lib
      )
      target_link_libraries(
        quantized_ops_aot_lib PUBLIC quantized_ops_pybind_lib
      )

      # pip wheels will need to be able to find the dependent libraries. On
      # Linux, the .so has non-absolute dependencies on libs like
      # "_portable_lib.so" without paths; as long as we `import torch` first,
      # those dependencies will work. But Apple dylibs do not support
      # non-absolute dependencies, so we need to tell the loader where to look
      # for its libraries. The LC_LOAD_DYLIB entries for the portable_lib
      # libraries will look like "@rpath/_portable_lib.cpython-310-darwin.so",
      # so we can add an LC_RPATH entry to look in a directory relative to the
      # installed location of our _portable_lib.so file. To see these LC_*
      # values, run `otool -l libquantized_ops_lib.dylib`.
      if(APPLE)
        set(RPATH "@loader_path/../../extensions/pybindings")
      else()
        set(RPATH "$ORIGIN/../../extensions/pybindings")
      endif()
      set_target_properties(
        quantized_ops_aot_lib PROPERTIES BUILD_RPATH ${RPATH} INSTALL_RPATH
                                                              ${RPATH}
      )
    endif()
  endif()
endif()

add_library(quantized_kernels ${_quantized_kernels__srcs})
target_link_libraries(
  quantized_kernels PRIVATE executorch_core kernels_util_all_deps
)
target_compile_options(quantized_kernels PUBLIC ${_common_compile_options})
# Build a library for _quantized_kernels_srcs
#
# quantized_ops_lib: Register quantized ops kernels into Executorch runtime
gen_operators_lib(
  LIB_NAME "quantized_ops_lib" KERNEL_LIBS quantized_kernels DEPS
  executorch_core
)

install(
  TARGETS quantized_kernels quantized_ops_lib
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/kernels/quantized/
)
