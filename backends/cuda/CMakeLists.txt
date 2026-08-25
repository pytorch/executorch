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

# Use dynamic linking for the GPU runtime.
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

if(EXECUTORCH_BUILD_ROCM)
  find_package(hip CONFIG REQUIRED)
  set(_cuda_runtime_libraries hip::host)
  set(_cuda_include_directories "")
else()
  find_package(CUDAToolkit REQUIRED)
  set(_cuda_runtime_libraries CUDA::cudart)
  set(_cuda_include_directories ${CUDAToolkit_INCLUDE_DIRS})
endif()

# Try to enable CUDA language when a working CUDA compiler toolchain is
# available. Some CI environments (notably Windows packaging jobs) provide
# CUDAToolkit headers/libs but cannot complete CUDA compiler identification. In
# those cases, keep configuration working and skip CUDA-only sources below.
if(NOT EXECUTORCH_BUILD_ROCM AND NOT CMAKE_CUDA_COMPILER)
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
if(EXECUTORCH_BUILD_ROCM)
  # ROCm never enables the CUDA language below, so there is no architecture list
  # to choose and nothing here would be read.
elseif(DEFINED ENV{TORCH_CUDA_ARCH_LIST})
  string(REPLACE "." "" _executorch_cuda_arch_list "$ENV{TORCH_CUDA_ARCH_LIST}")
  string(REPLACE " " ";" _executorch_cuda_arch_list
                 "${_executorch_cuda_arch_list}"
  )
  set(_executorch_cuda_arch_resolved "")
  set(_executorch_cuda_arch_newest "")
  set(_executorch_cuda_arch_newest_rank 0)
  # Mapped to CMake's explicit kinds rather than left bare. A bare number asks
  # nvcc for both the binary and the portable form, which measured at about 2.5x
  # the device code for a four architecture row. Only the newest needs the
  # portable form, since that is what lets a device newer than every entry here
  # run this code by compiling it on load. torch's own "+PTX" spelling of that
  # request is dropped on the way, because nvcc rejects the suffix literally and
  # the -virtual entry below already covers what it asks for.
  foreach(_arch IN LISTS _executorch_cuda_arch_list)
    string(REGEX REPLACE "\\+PTX$" "" _arch "${_arch}")
    list(APPEND _executorch_cuda_arch_resolved "${_arch}-real")
    # Compared numerically rather than taken as the last entry written. Nothing
    # orders TORCH_CUDA_ARCH_LIST, so a row spelled "9.0 8.0" put the portable
    # form on 8.0 and left a device newer than 9.0 with nothing to compile on
    # load. The trailing letters some architectures carry are dropped for the
    # comparison only, and a family name reduces to nothing and is skipped,
    # which is the same answer the numeric filter below reaches.
    string(REGEX MATCH "^[0-9]+" _executorch_cuda_arch_rank "${_arch}")
    if(_executorch_cuda_arch_rank AND _executorch_cuda_arch_rank GREATER
                                      _executorch_cuda_arch_newest_rank
    )
      set(_executorch_cuda_arch_newest_rank "${_executorch_cuda_arch_rank}")
      set(_executorch_cuda_arch_newest "${_arch}")
    endif()
  endforeach()
  if(_executorch_cuda_arch_newest)
    list(APPEND _executorch_cuda_arch_resolved
         "${_executorch_cuda_arch_newest}-virtual"
    )
  endif()
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
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  message(
    STATUS "CUDA architectures left to torch: ${CMAKE_CUDA_ARCHITECTURES}"
  )
endif()

if(NOT EXECUTORCH_BUILD_ROCM AND CMAKE_CUDA_COMPILER)
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
target_link_libraries(cuda_platform PRIVATE ${CMAKE_DL_LIBS})
if(EXECUTORCH_BUILD_SHARED)
  # Resolve the runtime from the shared library rather than from the static
  # core, for the same reason the backend below does: the shims library that
  # ships in the wheel links this one, so taking the static core here gives a
  # process that also loads the shipped runtime two platform layers, and a
  # register_pal call through the runtime never reaches the shims. Linking the
  # shared runtime leaves one platform layer the process has.
  target_link_libraries(cuda_platform PRIVATE executorch_shared)
else()
  target_link_libraries(cuda_platform PRIVATE executorch_core)
endif()

install(
  TARGETS cuda_platform
  EXPORT ExecuTorchTargets
  DESTINATION lib
)

# CUDA-specific AOTI shim symbols (dynamically linked)
set(_aoti_cuda_shim_sources runtime/cuda_allocator.cpp runtime/shims/memory.cpp
                            runtime/shims/cuda_guard.cpp
)

# Only build CUDA-only operator shims when the CUDA toolchain is available.
if(NOT EXECUTORCH_BUILD_ROCM AND CMAKE_CUDA_COMPILER)
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
if(EXECUTORCH_BUILD_ROCM)
  target_compile_definitions(aoti_cuda_shims PUBLIC EXECUTORCH_USE_HIP=1)
endif()

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
  PUBLIC ${_cuda_include_directories} $<BUILD_INTERFACE:${EXECUTORCH_ROOT}>
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
    aoti_cuda_shims PRIVATE cuda_platform ${_cuda_runtime_libraries}
                            extension_cuda ${CMAKE_DL_LIBS}
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
    PUBLIC ${_cuda_runtime_libraries} extension_cuda ${CMAKE_DL_LIBS}
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
# other call, because the helper is only meaningful for an install whose
# dependencies sit beside the target, which is what a shared build produces. It
# appends to any INSTALL_RPATH the caller set rather than replacing it, so a
# source install keeps its own paths.
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
                               runtime/cuda_weight_cache.cpp
)
if(_cuda_is_msvc_toolchain)
  # MSVC links aoti_cuda_backend into portable_lib without relying on C++
  # symbols exported from aoti_cuda_shims.dll.
  list(APPEND _aoti_cuda_backend_sources runtime/cuda_allocator.cpp)
endif()

# Build the delegate as a shared library for the wheel so a process has one copy
# of it, and keep it static everywhere else so no other build changes.
#
# ROCm is not one of the wheels. Packaging selects this file on the CUDA flag
# alone, so a shared library built here for a ROCm install is linked by the
# Python extension and then never copied beside it, and the installed extension
# fails to load on a missing dependency. Static is also what a ROCm build
# produced before the wheel layout existed, so nothing there changes.
if(EXECUTORCH_BUILD_SHARED AND NOT EXECUTORCH_BUILD_ROCM)
  set(_executorch_cuda_ships_libraries ON)
  set(_aoti_cuda_backend_library_type SHARED)
else()
  set(_executorch_cuda_ships_libraries OFF)
  set(_aoti_cuda_backend_library_type STATIC)
endif()
add_library(
  aoti_cuda_backend ${_aoti_cuda_backend_library_type}
                    ${_aoti_cuda_backend_sources}
)

target_include_directories(
  aoti_cuda_backend
  PUBLIC ${_cuda_include_directories} $<BUILD_INTERFACE:${EXECUTORCH_ROOT}>
         $<INSTALL_INTERFACE:include>
)
if(EXECUTORCH_BUILD_ROCM)
  target_compile_definitions(aoti_cuda_backend PUBLIC EXECUTORCH_USE_HIP=1)
endif()
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
  aoti_cuda_backend
  PUBLIC cuda_platform extension_tensor ${_cuda_runtime_libraries}
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
  # Resolve the runtime from the shared library rather than from the static
  # core, so the delegate registers into the one registry the process has. True
  # of a static delegate in a shared build too, which is what ROCm builds.
  target_link_libraries(aoti_cuda_backend PUBLIC executorch_shared)
endif()

if(_executorch_cuda_ships_libraries)
  # Named after what the library provides rather than after the target that
  # produces it, matching the other shipped delegates, so the file reads as
  # libexecutorch_backend_cuda.so. The target name stays as it is because the
  # rest of the build already refers to it.
  set_target_properties(
    aoti_cuda_backend PROPERTIES OUTPUT_NAME executorch_backend_cuda
  )
  executorch_target_soname_policy(aoti_cuda_backend)
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

  et_cxx_test(
    test_cuda_weight_cache SOURCES runtime/test/test_cuda_weight_cache.cpp
    EXTRA_LIBS aoti_cuda_backend
  )
  target_compile_definitions(test_cuda_weight_cache PRIVATE CUDA_AVAILABLE=1)
endif()
