# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Build AOTI CUDA backend for runtime.
#
# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#
cmake_minimum_required(VERSION 3.29)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

# Use dynamic linking for CUDA runtime
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

find_package(CUDAToolkit REQUIRED)

# Try to enable CUDA language when a working CUDA compiler toolchain is
# available. Some CI environments (notably Windows packaging jobs) provide
# CUDAToolkit headers/libs but cannot complete CUDA compiler identification. In
# those cases, keep configuration working and skip CUDA-only sources below.
if(NOT CMAKE_CUDA_COMPILER)
  include(CheckLanguage)
  check_language(CUDA)
endif()

# Take the architectures from the release row when it names them, before the
# language is enabled, since CMake fixes them at that point. Without this the
# build uses CMake's default, which on some devices is older than the intrinsics
# these sources use, and the compile fails with an undefined identifier that
# looks like a source problem.
#
# TORCH_CUDA_ARCH_LIST is the variable the surrounding build environment already
# sets, in PyTorch's dotted form. CMake wants bare integers, so "9.0" becomes
# 90. A "+PTX" suffix asks for the portable form in addition to the compiled
# one, which is what PyTorch means by it, so it adds the -virtual kind rather
# than replacing the -real one. Torch sets this to OFF at root scope when it is
# defined, warning that it ignores the value, so a defined value does not mean a
# caller chose it. OFF is treated as absent here, otherwise asking for an
# architecture through the preset silently compiles for whatever torch's own
# gencode flags select instead. CMAKE_CUDA_ARCHITECTURES is deliberately not
# read here. torch's CMake rejects and overrides it, which is why the release
# rows publish TORCH_CUDA_ARCH_LIST instead, and CMake fills the cache entry
# with a default of its own once the CUDA language is enabled. torch enables
# that language before this directory is added, so the cache always holds a
# value and cannot be read as a caller's intent: doing so replaced torch's
# autodetected architecture with CMake's default and broke the build on a device
# the default does not cover.
if(DEFINED ENV{TORCH_CUDA_ARCH_LIST})
  string(REPLACE "." "" _executorch_cuda_arch_list "$ENV{TORCH_CUDA_ARCH_LIST}")
  string(REPLACE " " ";" _executorch_cuda_arch_list
                 "${_executorch_cuda_arch_list}"
  )
  # torch spells a request for the portable form as a "+PTX" suffix, which nvcc
  # rejects literally. Drop the suffix rather than expanding it into a separate
  # -virtual entry: an unsuffixed CMake architecture already asks for both the
  # compiled and the portable form, measured as code=[compute_120,sm_120] for a
  # bare "120", so the extra entry only duplicates a gencode that is already
  # there.
  set(_executorch_cuda_arch_resolved "")
  foreach(_arch IN LISTS _executorch_cuda_arch_list)
    string(REGEX REPLACE "\\+PTX$" "" _arch "${_arch}")
    list(APPEND _executorch_cuda_arch_resolved "${_arch}")
  endforeach()
  # A row that names both "12.0" and "12.0+PTX" collapses to the same entry
  # twice, which would duplicate a gencode on the compile line.
  list(REMOVE_DUPLICATES _executorch_cuda_arch_resolved)
  # torch also accepts family names such as Ampere or Hopper, which its own
  # select_compute_arch.cmake documents and has already turned into gencode
  # flags by the time this directory is added. Dropped rather than rejected:
  # failing here would break a configuration that builds today, and passing a
  # name through to CMAKE_CUDA_ARCHITECTURES would reach nvcc, which only
  # accepts numbers.
  set(_executorch_cuda_arch_numeric)
  foreach(_arch IN LISTS _executorch_cuda_arch_resolved)
    if(_arch MATCHES "^[0-9]+[a-z]*(-real|-virtual)?$")
      list(APPEND _executorch_cuda_arch_numeric "${_arch}")
    else()
      message(
        STATUS
          "executorch: leaving TORCH_CUDA_ARCH_LIST entry \"${_arch}\" to torch, since it names a "
          "GPU family rather than a compute capability"
      )
    endif()
  endforeach()
  # Left unset when nothing numeric remains, so torch's resolution is what
  # decides. Setting it to an empty string would instead ask CMake to compile
  # for no architecture at all.
  if(_executorch_cuda_arch_numeric)
    set(CMAKE_CUDA_ARCHITECTURES "${_executorch_cuda_arch_numeric}")
  endif()
  message(
    STATUS
      "CUDA architectures from TORCH_CUDA_ARCH_LIST: ${CMAKE_CUDA_ARCHITECTURES}"
  )
else()
  # torch sets this to OFF on purpose and drives nvcc with its own gencode
  # flags, so there is nothing to choose here and overriding it would replace a
  # working set of architectures with one value.
  #
  # An empty value is a different case: nothing selected the architectures, and
  # an empty list makes nvcc fall back to its built-in compute_50, which current
  # CUDA rejects outright. Ask CMake to detect the builder's GPU instead, which
  # is what a source build wants.
  if("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
    # An empty list makes nvcc fall back to its own default, which current CUDA
    # rejects outright. Detection is not the answer either: on a machine without
    # a usable device it resolves to an architecture nvcc already warns it will
    # stop supporting. Name the two datacentre architectures the release rows
    # build for, so a source build produces device code that runs on the same
    # GPUs the wheels target.
    set(CMAKE_CUDA_ARCHITECTURES "80;86")
  endif()
  message(
    STATUS "CUDA architectures left to torch: ${CMAKE_CUDA_ARCHITECTURES}"
  )
endif()

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
endif()

# Centralize Windows/MSVC checks used throughout this file.
set(_cuda_is_msvc_toolchain OFF)
if(MSVC)
  set(_cuda_is_msvc_toolchain ON)
endif()

set(_cuda_is_windows_msvc OFF)
if(WIN32 AND _cuda_is_msvc_toolchain)
  set(_cuda_is_windows_msvc ON)
endif()

# Common C++ compile options for CUDA backend targets.
if(_cuda_is_msvc_toolchain)
  set(_cuda_cxx_compile_options /EHsc /GR)
else()
  set(_cuda_cxx_compile_options -fexceptions -frtti -fPIC)
endif()

# Platform-specific linker option for exporting symbols from shared libs.
set(_cuda_export_dynamic_option "")
if(NOT _cuda_is_msvc_toolchain)
  if(APPLE)
    set(_cuda_export_dynamic_option -Wl,-export_dynamic)
  else()
    set(_cuda_export_dynamic_option -Wl,--export-dynamic)
  endif()
endif()

# Use ExecuTorch's standard way to find PyTorch libraries for AOTI
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
find_package_torch()

# Platform utilities (load_library, close_library, etc.)
set(_cuda_platform_sources runtime/platform/platform.cpp)
add_library(cuda_platform STATIC ${_cuda_platform_sources})

target_include_directories(
  cuda_platform
  PUBLIC $<BUILD_INTERFACE:${EXECUTORCH_ROOT}> $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
)

target_compile_options(
  cuda_platform
  PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${_cuda_cxx_compile_options}>"
)

# Link against ExecuTorch core libraries
target_link_libraries(cuda_platform PRIVATE executorch_core ${CMAKE_DL_LIBS})

install(
  TARGETS cuda_platform
  EXPORT ExecuTorchTargets
  DESTINATION lib
)

# CUDA-specific AOTI shim symbols (dynamically linked)
set(_aoti_cuda_shim_sources runtime/cuda_allocator.cpp runtime/shims/memory.cpp
                            runtime/shims/cuda_guard.cpp
)

# Only build CUDA shims when CUDA language/toolchain is available.
if(CMAKE_CUDA_COMPILER)
  list(
    APPEND
    _aoti_cuda_shim_sources
    runtime/shims/int4mm.cu
    runtime/shims/int4_plain_mm.cu
    runtime/shims/int5_plain_mm.cu
    runtime/shims/int6_plain_mm.cu
    runtime/shims/int8_plain_mm.cu
    runtime/shims/sort.cu
    runtime/shims/rand.cu
  )
endif()

add_library(aoti_cuda_shims SHARED ${_aoti_cuda_shim_sources})

# Define CUDA_AVAILABLE to use SlimTensor on GPU in common_shims_slim.h
target_compile_definitions(aoti_cuda_shims PRIVATE CUDA_AVAILABLE=1)

# Define export macros for shared library. Use WIN32 (not just MSVC) so MinGW
# cross-compiles also emit dllexport symbols for AOTI shims.
if(WIN32)
  target_compile_definitions(aoti_cuda_shims PRIVATE EXPORT_AOTI_FUNCTIONS)
  if(_cuda_is_windows_msvc)
    # Ensure proper DLL import/export library naming on Windows
    set_target_properties(
      aoti_cuda_shims PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS OFF
    )
  endif()
endif()

target_include_directories(
  aoti_cuda_shims
  PUBLIC ${CUDAToolkit_INCLUDE_DIRS} $<BUILD_INTERFACE:${EXECUTORCH_ROOT}>
         $<INSTALL_INTERFACE:include>
)

target_compile_options(
  aoti_cuda_shims
  PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${_cuda_cxx_compile_options}>"
)

# Ensure symbols are exported properly
if(_cuda_export_dynamic_option)
  target_link_options(aoti_cuda_shims PUBLIC ${_cuda_export_dynamic_option})
endif()

# Link against CUDA::cudart, common AOTI library, and platform utilities. On
# non-MSVC, use --whole-archive for aoti_common_shims_slim to force shim symbol
# retention.
if(_cuda_is_msvc_toolchain)
  target_link_libraries(
    aoti_cuda_shims PRIVATE cuda_platform CUDA::cudart extension_cuda
                            ${CMAKE_DL_LIBS}
  )
  # Link object library directly so symbols are pulled exactly once while
  # avoiding duplicate static/object inclusion and interface leakage.
  target_link_libraries(aoti_cuda_shims PRIVATE aoti_common_shims_slim_obj)
else()
  # The whole-archive pair is PRIVATE so it applies to this library's own link
  # and does not replay at the link of anything that links this one. In PUBLIC
  # scope the archive was extracted again by each consumer, so the translation
  # unit holding SlimTensor's per-device current stream thread_local landed in
  # three shipped binaries, and one copy per process is what makes that state
  # agree. The MSVC branch above uses PRIVATE for the same reason.
  target_link_libraries(
    aoti_cuda_shims
    PRIVATE cuda_platform -Wl,--whole-archive aoti_common_shims_slim
            -Wl,--no-whole-archive
    PUBLIC CUDA::cudart extension_cuda ${CMAKE_DL_LIBS}
  )
endif()

if(NOT _cuda_is_msvc_toolchain)
  executorch_target_link_options_shared_lib(aoti_cuda_shims)
endif()

# This library links the CUDA runtime directly and the wheel ships it, so it
# needs a search path for the same reason the delegate does. Without one it can
# ship with no runtime path at all, on a builder where the runtime resolves from
# an implicit link directory, and the wheel then adds no relative hop either,
# because the step that adds one has nothing to rewrite. Guarded like every
# other call, because the helper replaces INSTALL_RPATH and a source install may
# have been given one that points at its own CUDA runtime.
if(EXECUTORCH_BUILD_SHARED)
  executorch_target_shipped_runtime_path(aoti_cuda_shims)
endif()

install(
  TARGETS aoti_cuda_shims
  EXPORT ExecuTorchTargets
  DESTINATION lib
)

# CUDA backend implementation
set(_aoti_cuda_backend_sources runtime/cuda_backend.cpp
                               runtime/cuda_mutable_state.cpp
)
if(_cuda_is_msvc_toolchain)
  # MSVC links aoti_cuda_backend into portable_lib without relying on C++
  # symbols exported from aoti_cuda_shims.dll.
  list(APPEND _aoti_cuda_backend_sources runtime/cuda_allocator.cpp)
endif()

# Build the delegate as a shared library for the wheel so a process has one copy
# of it, and keep it static everywhere else so no other build changes.
if(EXECUTORCH_BUILD_SHARED)
  set(_aoti_cuda_backend_library_type SHARED)
else()
  set(_aoti_cuda_backend_library_type STATIC)
endif()
add_library(
  aoti_cuda_backend ${_aoti_cuda_backend_library_type}
                    ${_aoti_cuda_backend_sources}
)

target_include_directories(
  aoti_cuda_backend
  PUBLIC ${CUDAToolkit_INCLUDE_DIRS} $<BUILD_INTERFACE:${EXECUTORCH_ROOT}>
         $<INSTALL_INTERFACE:include>
)
target_compile_options(
  aoti_cuda_backend
  PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${_cuda_cxx_compile_options}>"
)
# Ensure symbols are exported properly
if(_cuda_export_dynamic_option)
  target_link_options(aoti_cuda_backend PUBLIC ${_cuda_export_dynamic_option})
endif()

# Link against shims library and other dependencies On Windows (MSVC), use
# PRIVATE linkage for aoti_cuda_shims since the DLL is copied to the executable
# directory. On other platforms, use PUBLIC so the dependency propagates to
# consumers.
target_link_libraries(
  aoti_cuda_backend PUBLIC cuda_platform extension_tensor CUDA::cudart
                           extension_cuda ${CMAKE_DL_LIBS}
)

if(_cuda_is_msvc_toolchain)
  # cuda_backend.cpp uses SlimTensor CUDA utilities (e.g. getCurrentCUDAStream)
  # from aoti_common_shims_slim via headers; propagate the static lib so final
  # MSVC links (e.g. parakeet_runner) can resolve those C++ symbols.
  target_link_libraries(
    aoti_cuda_backend
    PRIVATE aoti_cuda_shims
    PUBLIC aoti_common_shims_slim
  )
else()
  target_link_libraries(aoti_cuda_backend PUBLIC aoti_cuda_shims)
endif()

executorch_target_link_options_shared_lib(aoti_cuda_backend)

if(EXECUTORCH_BUILD_SHARED)
  # Named after what the library provides rather than after the target that
  # produces it, matching the other shipped delegates, so the file reads as
  # libexecutorch_backend_cuda.so. The target name stays as it is because the
  # rest of the build already refers to it.
  set_target_properties(
    aoti_cuda_backend PROPERTIES OUTPUT_NAME executorch_backend_cuda
  )
  executorch_target_soname_policy(aoti_cuda_backend)
  # Resolve the runtime from the shared library rather than from the static
  # core, so the delegate registers into the one registry the process has.
  target_link_libraries(aoti_cuda_backend PUBLIC executorch_shared)
  # Ships beside the runtime in the wheel's lib/ directory.
  executorch_target_shipped_runtime_path(aoti_cuda_backend)
endif()

install(
  TARGETS aoti_cuda_backend
  EXPORT ExecuTorchTargets
  DESTINATION lib
)

if(BUILD_TESTING)
  include(${EXECUTORCH_ROOT}/tools/cmake/Test.cmake)

  et_cxx_test(
    test_cuda_allocator SOURCES runtime/test/test_cuda_allocator.cpp EXTRA_LIBS
    aoti_cuda_backend
  )
  target_compile_definitions(test_cuda_allocator PRIVATE CUDA_AVAILABLE=1)

  et_cxx_test(
    test_cuda_mutable_state SOURCES runtime/test/test_cuda_mutable_state.cpp
    EXTRA_LIBS aoti_cuda_backend
  )
  target_compile_definitions(test_cuda_mutable_state PRIVATE CUDA_AVAILABLE=1)
endif()
