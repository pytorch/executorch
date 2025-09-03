# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
add_subdirectory(json)
add_subdirectory(gflags)

if(EXECUTORCH_BUILD_PYBIND)
  add_subdirectory(pybind11)
endif()

if(BUILD_TESTING)
  add_subdirectory(googletest)
endif()

# MARK: - flatbuffers

if(WIN32)
  set(_executorch_external_project_additional_args)
else()
  # Always use Make to avoid needing to codesign flatc if the project is using Xcode.
  set(_executorch_external_project_additional_args CMAKE_GENERATOR "Unix Makefiles")
endif()

# We use ExternalProject to build flatc from source to force it target the host.
# Otherwise, flatc will target the project's toolchain (i.e. iOS, or Android).
ExternalProject_Add(
  flatbuffers_ep
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/flatc_proj
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/third-party/flatbuffers
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/flatc_proj/src/build
  CMAKE_ARGS -DFLATBUFFERS_BUILD_FLATC=ON
             -DFLATBUFFERS_INSTALL=ON
             -DFLATBUFFERS_BUILD_FLATHASH=OFF
             -DFLATBUFFERS_BUILD_FLATLIB=OFF
             -DFLATBUFFERS_BUILD_TESTS=OFF
             -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
             -DCMAKE_CXX_FLAGS="-DFLATBUFFERS_MAX_ALIGNMENT=${EXECUTORCH_FLATBUFFERS_MAX_ALIGNMENT}"
             # Unset the toolchain to build for the host instead of the toolchain set for the project.
             -DCMAKE_TOOLCHAIN_FILE=
             # If building for iOS, "unset" these variables to rely on the host (macOS) defaults.
             $<$<AND:$<BOOL:${APPLE}>,$<BOOL:$<FILTER:${PLATFORM},EXCLUDE,^MAC>>>:-DCMAKE_OSX_SYSROOT=>
             -DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=${CMAKE_OSX_DEPLOYMENT_TARGET}
  BUILD_BYPRODUCTS <INSTALL_DIR>/bin/flatc
  ${_executorch_external_project_additional_args}
)
ExternalProject_Get_Property(flatbuffers_ep INSTALL_DIR)
add_executable(flatc IMPORTED GLOBAL)
add_dependencies(flatc flatbuffers_ep)
if(WIN32 AND NOT CMAKE_CROSSCOMPILING)
  # flatbuffers does not use CMAKE_BUILD_TYPE. Internally, the build forces Release
  # config, but from CMake's perspective the build type is always Debug.
  set_target_properties(flatc PROPERTIES IMPORTED_LOCATION ${INSTALL_DIR}/bin/flatc.exe)
else()
  set_target_properties(flatc PROPERTIES IMPORTED_LOCATION ${INSTALL_DIR}/bin/flatc)
endif()

# TODO: re-enable once flatbuffers is added as a subdirectory.
# set(FLATBUFFERS_BUILD_FLATC OFF)
# set(FLATBUFFERS_INSTALL OFF)
# set(FLATBUFFERS_BUILD_FLATHASH OFF)
# set(FLATBUFFERS_BUILD_FLATLIB OFF)
# set(FLATBUFFERS_BUILD_TESTS OFF)

# MARK: - flatcc

if(WIN32)
  # For some reason, when configuring the external project during build
  # CMAKE_C_SIMULATE_ID is set to MSVC, but CMAKE_CXX_SIMULATE_ID is not set.
  # To make sure the external project is configured correctly, set it explicitly
  # here.
  set(_flatcc_extra_cmake_args -DCMAKE_CXX_SIMULATE_ID=MSVC)
else()
  set(_flatcc_extra_cmake_args)
endif()

# Similar to flatbuffers, we want to build flatcc for the host. See inline comments
# in the flatbuffers ExternalProject_Add for more details.
ExternalProject_Add(
  flatcc_ep
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/flatcc_ep
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/third-party/flatcc
  CMAKE_ARGS -DFLATCC_RTONLY=OFF
             -DFLATCC_TEST=OFF
             -DFLATCC_REFLECTION=OFF
             -DFLATCC_DEBUG_CLANG_SANITIZE=OFF
             -DFLATCC_INSTALL=ON
             -DCMAKE_POLICY_VERSION_MINIMUM=3.5
             -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
             -DCMAKE_TOOLCHAIN_FILE=
             $<$<AND:$<BOOL:${APPLE}>,$<BOOL:$<FILTER:${PLATFORM},EXCLUDE,^MAC>>>:-DCMAKE_OSX_SYSROOT=>
             -DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=${CMAKE_OSX_DEPLOYMENT_TARGET}
             ${_flatcc_extra_cmake_args}
  BUILD_BYPRODUCTS <INSTALL_DIR>/bin/flatcc
  {_executorch_external_project_additional_args}
)
file(REMOVE_RECURSE ${PROJECT_SOURCE_DIR}/third-party/flatcc/lib)
ExternalProject_Get_Property(flatcc_ep INSTALL_DIR)
add_executable(flatcc_cli IMPORTED GLOBAL)
add_dependencies(flatcc_cli flatcc_ep)
if(WIN32 AND NOT CMAKE_CROSSCOMPILING)
  set_target_properties(flatcc_cli PROPERTIES IMPORTED_LOCATION ${INSTALL_DIR}/bin/flatcc.exe)
else()
  set_target_properties(flatcc_cli PROPERTIES IMPORTED_LOCATION ${INSTALL_DIR}/bin/flatcc)
endif()

set(FLATCC_RTONLY ON CACHE BOOL "")
set(FLATCC_TEST OFF CACHE BOOL "")
set(FLATCC_REFLECTION OFF CACHE BOOL "")
set(FLATCC_DEBUG_CLANG_SANITIZE OFF CACHE BOOL "")
set(FLATCC_INSTALL OFF CACHE BOOL "")
add_subdirectory(flatcc)
# Unfortunately flatcc writes libs directly in to the source tree [1]. So to
# ensure the target lib is created last, force flatcc_cli to build first.
#
# [1] https://github.com/dvidelabs/flatcc/blob/896db54787e8b730a6be482c69324751f3f5f117/CMakeLists.txt#L168
add_dependencies(flatccrt flatcc_cli)
# Fix for "relocation R_X86_64_32 against `.rodata' can not be used when making
# a shared object; recompile with -fPIC" when building on some x86 linux
# systems.
#
# Learn more: https://github.com/pytorch/executorch/pull/2467
set_property(TARGET flatccrt PROPERTY POSITION_INDEPENDENT_CODE ON)
install(
  TARGETS flatccrt
  DESTINATION ${CMAKE_BINARY_DIR}/lib
)
