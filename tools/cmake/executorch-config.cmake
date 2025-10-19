# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Config defining how CMake should find ExecuTorch package. CMake will search
# for this file and find ExecuTorch package if it is installed. Typical usage
# is:
#
# find_package(executorch REQUIRED)
# -------
#
# Finds the ExecuTorch library
#
# This will define the following variables:
#
# EXECUTORCH_FOUND        -- True if the system has the ExecuTorch library
# EXECUTORCH_INCLUDE_DIRS -- The include directories for ExecuTorch
# EXECUTORCH_LIBRARIES    -- Libraries to link against
#
# The actual values for these variables will be different from what
# executorch-config.cmake in executorch pip package gives, but we wanted to keep
# the contract of exposing these CMake variables.

cmake_minimum_required(VERSION 3.24)

include(CMakeFindDependencyMacro)
find_package(tokenizers CONFIG)

set(_root "${CMAKE_CURRENT_LIST_DIR}/../../..")
set(required_lib_list executorch executorch_core portable_kernels)
set(EXECUTORCH_LIBRARIES)
set(EXECUTORCH_INCLUDE_DIRS
    ${_root}/include ${_root}/include/executorch/runtime/core/portable_type/c10
    ${_root}/lib
)
foreach(lib ${required_lib_list})
  set(lib_var "LIB_${lib}")
  find_library(
    ${lib_var} ${lib}
    HINTS "${_root}/lib"
    CMAKE_FIND_ROOT_PATH_BOTH
  )
  if(NOT ${lib_var})
    set(EXECUTORCH_FOUND OFF)
    return()
  endif()
  list(APPEND EXECUTORCH_LIBRARIES ${lib})
endforeach()
set(EXECUTORCH_FOUND ON)

include("${CMAKE_CURRENT_LIST_DIR}/ExecuTorchTargets.cmake")

set(optional_lib_list
    aoti_cuda
    flatccrt
    etdump
    bundled_program
    extension_data_loader
    extension_flat_tensor
    coreml_util
    coreml_inmemoryfs
    coremldelegate
    mpsdelegate
    neuron_backend
    qnn_executorch_backend
    portable_ops_lib
    custom_ops
    extension_evalue_util
    extension_llm_runner
    extension_module
    extension_module_static
    extension_runner_util
    extension_tensor
    extension_threadpool
    extension_training
    xnnpack_backend
    vulkan_backend
    optimized_kernels
    optimized_portable_kernels
    cpublas
    eigen_blas
    optimized_ops_lib
    optimized_native_cpu_ops_lib
    quantized_kernels
    quantized_ops_lib
    quantized_ops_aot_lib
    openvino_backend
    torchao_ops_executorch
    torchao_kernels_aarch64
)

foreach(lib ${optional_lib_list})
  if(TARGET ${lib})
    list(APPEND EXECUTORCH_LIBRARIES ${lib})
  else()
    message("${lib} library is not found.
             If needed rebuild with the proper options in CMakeLists.txt"
    )
  endif()
endforeach()

# The ARM baremetal size test's CMAKE_TOOLCHAIN_FILE apparently doesn't prevent
# our attempts to find_library(dl) from succeeding when building ExecuTorch, but
# that call finds the host system's libdl and there is no actual libdl available
# when building for the actual final baremetal.
get_property(
  FIXED_EXECUTORCH_CORE_LINK_LIBRARIES
  TARGET executorch_core
  PROPERTY INTERFACE_LINK_LIBRARIES
)
list(REMOVE_ITEM FIXED_EXECUTORCH_CORE_LINK_LIBRARIES $<LINK_ONLY:dl>)
set_property(
  TARGET executorch_core PROPERTY INTERFACE_LINK_LIBRARIES
                                  ${FIXED_EXECUTORCH_CORE_LINK_LIBRARIES}
)
