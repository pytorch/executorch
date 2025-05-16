# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# MARK: - flatbuffers

# We use ExternalProject to build flatc from source to force it target the host.
# Otherwise, flatc will target the project's toolchain (i.e. iOS, or Android).
ExternalProject_Add(
  flatbuffers_external_project
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers_external_project
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/third-party/flatbuffers
  # Always use Make to avoid needing to codesign flatc if the project is using Xcode.
  CMAKE_GENERATOR "Unix Makefiles"
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
)
ExternalProject_Get_Property(flatbuffers_external_project INSTALL_DIR)
add_executable(flatc IMPORTED GLOBAL)
add_dependencies(flatc flatbuffers_external_project)
if(WIN32)
  # flatbuffers does not use CMAKE_BUILD_TYPE. Internally, the build forces Release
  # config, but from CMake's perspective the build type is always Debug.
  set_target_properties(flatc PROPERTIES IMPORTED_LOCATION ${INSTALL_DIR}/$<CONFIG>/bin/flatc.exe)
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

# Similar to flatbuffers, we want to build flatcc for the host. See inline comments
# in the flatbuffers ExternalProject_Add for more details.
ExternalProject_Add(
  flatcc_external_project
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/flatcc_external_project
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/third-party/flatcc
  CMAKE_GENERATOR "Unix Makefiles"
  CMAKE_ARGS -DFLATCC_RTONLY=OFF
             -DFLATCC_TEST=OFF
             -DFLATCC_REFLECTION=OFF
             -DFLATCC_DEBUG_CLANG_SANITIZE=OFF
             -DFLATCC_INSTALL=ON
             -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
             -DCMAKE_TOOLCHAIN_FILE=
             $<$<AND:$<BOOL:${APPLE}>,$<BOOL:$<FILTER:${PLATFORM},EXCLUDE,^MAC>>>:-DCMAKE_OSX_SYSROOT=>
             -DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=${CMAKE_OSX_DEPLOYMENT_TARGET}
  BUILD_BYPRODUCTS <INSTALL_DIR>/bin/flatcc
)
ExternalProject_Get_Property(flatcc_external_project INSTALL_DIR)
add_executable(flatcc_cli IMPORTED GLOBAL)
add_dependencies(flatcc_cli flatcc_external_project)
if(WIN32)
  set_target_properties(flatcc_cli PROPERTIES IMPORTED_LOCATION ${INSTALL_DIR}/$<CONFIG>/bin/flatcc.exe)
else()
  set_target_properties(flatcc_cli PROPERTIES IMPORTED_LOCATION ${INSTALL_DIR}/bin/flatcc)
endif()

set(FLATCC_RTONLY ON CACHE BOOL "")
set(FLATCC_TEST OFF CACHE BOOL "")
set(FLATCC_REFLECTION OFF CACHE BOOL "")
set(FLATCC_DEBUG_CLANG_SANITIZE OFF CACHE BOOL "")
set(FLATCC_INSTALL OFF CACHE BOOL "")
add_subdirectory(flatcc)
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
