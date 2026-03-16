# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Kernel library for portable kernels. Please this file formatted by running:
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

set(_common_compile_options
    $<$<CXX_COMPILER_ID:MSVC>:/wd4996>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wno-deprecated-declarations>
)

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)

# Portable kernel sources TODO(larryliu0820): use buck2 to gather the sources
file(GLOB_RECURSE _portable_kernels__srcs
     "${CMAKE_CURRENT_SOURCE_DIR}/cpu/*.cpp"
)
list(FILTER _portable_kernels__srcs EXCLUDE REGEX "test/*.cpp")
list(FILTER _portable_kernels__srcs EXCLUDE REGEX "codegen")
# Generate C++ bindings to register kernels into both PyTorch (for AOT) and
# Executorch (for runtime). Here select all ops in functions.yaml
set(_yaml "${CMAKE_CURRENT_SOURCE_DIR}/functions.yaml")
gen_selected_ops(LIB_NAME "portable_ops_lib" OPS_SCHEMA_YAML "${_yaml}")
# Expect gen_selected_ops output file to be selected_operators.yaml
generate_bindings_for_kernels(
  LIB_NAME "portable_ops_lib" FUNCTIONS_YAML "${_yaml}"
)
message("Generated files ${gen_command_sources}")

#
# portable_kernels: Pure-C++ kernel library for ATen ops
#
# Focused on portability and understandability rather than speed.
#
add_library(portable_kernels ${_portable_kernels__srcs})
target_link_libraries(
  portable_kernels PRIVATE executorch_core kernels_util_all_deps
)
target_compile_options(portable_kernels PUBLIC ${_common_compile_options})

# Build a library for _portable_kernels__srcs
#
# portable_ops_lib: Register portable_ops_lib ops kernels into Executorch
# runtime
gen_operators_lib(
  LIB_NAME "portable_ops_lib" KERNEL_LIBS portable_kernels DEPS executorch_core
)

# Portable kernels support optional parallelization (and, in the future, perhaps
# other performance features). If support is present, produce an optimized
# version.
if(EXECUTORCH_BUILD_PTHREADPOOL AND EXECUTORCH_BUILD_KERNELS_OPTIMIZED)
  add_library(optimized_portable_kernels ${_portable_kernels__srcs})
  target_link_libraries(optimized_portable_kernels PRIVATE executorch_core)
  target_link_libraries(optimized_portable_kernels PUBLIC extension_threadpool)
  target_compile_options(
    optimized_portable_kernels PUBLIC ${_common_compile_options}
  )
  target_include_directories(
    optimized_portable_kernels PRIVATE ${TORCH_INCLUDE_DIRS}
  )
  target_compile_definitions(
    optimized_portable_kernels
    PRIVATE "ET_USE_PYTORCH_HEADERS=ET_HAS_EXCEPTIONS"
  )
  gen_selected_ops(
    LIB_NAME "optimized_portable_ops_lib" OPS_SCHEMA_YAML "${_yaml}"
  )
  generate_bindings_for_kernels(
    LIB_NAME "optimized_portable_ops_lib" FUNCTIONS_YAML "${_yaml}"
  )
  gen_operators_lib(
    LIB_NAME "optimized_portable_ops_lib" KERNEL_LIBS
    optimized_portable_kernels DEPS executorch_core
  )
  install(
    TARGETS optimized_portable_kernels optimized_portable_ops_lib
    EXPORT ExecuTorchTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()

install(
  TARGETS portable_kernels portable_ops_lib
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/kernels/portable/
)

# Build the portable custom ops AOT library for registering custom ops into
# PyTorch. Requires find_package(Torch), which must be called at root scope
# before this subdirectory is processed. Not targeting ARM_BAREMETAL as aot_lib
# depends on incompatible libraries
if(EXECUTORCH_BUILD_KERNELS_CUSTOM_AOT AND NOT EXECUTORCH_BUILD_ARM_BAREMETAL)
  set(_custom_ops_yaml "${CMAKE_CURRENT_SOURCE_DIR}/custom_ops.yaml")
  set(_portable_custom_ops "aten::allclose.out" "aten::allclose.Tensor")
  gen_selected_ops(
    LIB_NAME "portable_custom_ops_aot_lib" ROOT_OPS ${_portable_custom_ops}
  )
  generate_bindings_for_kernels(
    LIB_NAME "portable_custom_ops_aot_lib" CUSTOM_OPS_YAML
    "${_custom_ops_yaml}"
  )
  set(_portable_custom_ops_sources
      "${CMAKE_CURRENT_SOURCE_DIR}/cpu/op_allclose.cpp"
      "${EXECUTORCH_ROOT}/runtime/core/exec_aten/util/tensor_util_aten.cpp"
  )
  gen_custom_ops_aot_lib(
    LIB_NAME "portable_custom_ops_aot_lib" KERNEL_SOURCES
    ${_portable_custom_ops_sources}
  )
  target_include_directories(
    portable_custom_ops_aot_lib PRIVATE ${_common_include_directories}
  )

  if(APPLE)
    set(RPATH "@loader_path/../../extensions/pybindings")
  else()
    set(RPATH "\$ORIGIN/../../extensions/pybindings")
  endif()
  set_target_properties(
    portable_custom_ops_aot_lib PROPERTIES BUILD_RPATH ${RPATH} INSTALL_RPATH
                                                                ${RPATH}
  )
endif()
