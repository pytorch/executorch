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

cmake_minimum_required(VERSION 3.19)

set(_root "${CMAKE_CURRENT_LIST_DIR}/../..")
set(required_lib_list executorch executorch_no_prim_ops portable_kernels)
foreach(lib ${required_lib_list})
  set(lib_var "LIB_${lib}")
  add_library(${lib} STATIC IMPORTED)
  find_library(
    ${lib_var} ${lib}
    HINTS "${_root}"
    CMAKE_FIND_ROOT_PATH_BOTH
  )
  set_target_properties(${lib} PROPERTIES IMPORTED_LOCATION "${${lib_var}}")
  target_include_directories(${lib} INTERFACE ${_root})
endforeach()

target_link_libraries(executorch INTERFACE executorch_no_prim_ops)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(FLATCCRT_LIB flatccrt_d)
else()
  set(FLATCCRT_LIB flatccrt)
endif()

set(lib_list
    etdump
    bundled_program
    extension_data_loader
    ${FLATCCRT_LIB}
    coremldelegate
    mpsdelegate
    qnn_executorch_backend
    portable_ops_lib
    extension_module
    extension_module_static
    extension_runner_util
    extension_tensor
    extension_threadpool
    xnnpack_backend
    XNNPACK
    cpuinfo
    pthreadpool
    vulkan_backend
    optimized_kernels
    cpublas
    eigen_blas
    optimized_ops_lib
    optimized_native_cpu_ops_lib
    quantized_kernels
    quantized_ops_lib
    quantized_ops_aot_lib
)
foreach(lib ${lib_list})
  # Name of the variable which stores result of the find_library search
  set(lib_var "LIB_${lib}")
  find_library(
    ${lib_var} ${lib}
    HINTS "${_root}"
    CMAKE_FIND_ROOT_PATH_BOTH
  )
  if(NOT ${lib_var})
    message("${lib} library is not found.
            If needed rebuild with the proper options in CMakeLists.txt"
    )
  else()
    if("${lib}" STREQUAL "extension_module" AND (NOT CMAKE_TOOLCHAIN_IOS))
      add_library(${lib} SHARED IMPORTED)
    else()
      # Building a share library on iOS requires code signing, so it's easier to
      # keep all libs as static when CMAKE_TOOLCHAIN_IOS is used
      add_library(${lib} STATIC IMPORTED)
    endif()
    set_target_properties(${lib} PROPERTIES IMPORTED_LOCATION "${${lib_var}}")
    target_include_directories(${lib} INTERFACE ${_root})
  endif()
endforeach()
