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
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers_external_project
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/third-party/flatbuffers
  # Always use Make to avoid needing to codesign flatc if the project is using Xcode.
  CMAKE_GENERATOR "Unix Makefiles"
  CMAKE_ARGS -DFLATBUFFERS_BUILD_FLATC=ON
             -DFLATBUFFERS_INSTALL=OFF
             -DFLATBUFFERS_BUILD_FLATHASH=OFF
             -DFLATBUFFERS_BUILD_FLATLIB=OFF
             -DFLATBUFFERS_BUILD_TESTS=OFF
             -DCMAKE_CXX_FLAGS="-DFLATBUFFERS_MAX_ALIGNMENT=${EXECUTORCH_FLATBUFFERS_MAX_ALIGNMENT}"
             # Unset the toolchain to build for the host instead of the toolchain set for the project.
             -DCMAKE_TOOLCHAIN_FILE=
             # If building for iOS, "unset" these variables to rely on the host (macOS) defaults.
             $<$<AND:$<BOOL:${APPLE}>,$<BOOL:$<FILTER:${PLATFORM},EXCLUDE,^MAC>>>:-DCMAKE_OSX_SYSROOT=>
             -DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=${CMAKE_OSX_DEPLOYMENT_TARGET}
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS <BINARY_DIR>/flatc
)
ExternalProject_Get_Property(flatbuffers_external_project BINARY_DIR)
add_executable(flatc IMPORTED GLOBAL)
add_dependencies(flatc flatbuffers_external_project)
if(WIN32)
  # flatbuffers does not use CMAKE_BUILD_TYPE. Internally, the build forces Release
  # config, but from CMake's perspective the build type is always Debug.
  set_target_properties(flatc PROPERTIES IMPORTED_LOCATION ${BINARY_DIR}/$<CONFIG>/flatc.exe)
else()
  set_target_properties(flatc PROPERTIES IMPORTED_LOCATION ${BINARY_DIR}/flatc)
endif()

# TODO: re-enable once flatbuffers is added as a subdirectory.
# set(FLATBUFFERS_BUILD_FLATC OFF)
# set(FLATBUFFERS_INSTALL OFF)
# set(FLATBUFFERS_BUILD_FLATHASH OFF)
# set(FLATBUFFERS_BUILD_FLATLIB OFF)
# set(FLATBUFFERS_BUILD_TESTS OFF)
