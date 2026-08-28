# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

# Set minimum required CMake version
cmake_minimum_required(VERSION 3.19)

# Set project name
project(openvino_backend_project)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure compile_commands.json is generated
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set up EXECUTORCH_ROOT if not already set
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

# Include utility CMake scripts from ExecuteTorch
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
executorch_get_build_include_dir(
  "${EXECUTORCH_ROOT}" EXECUTORCH_SOURCE_INCLUDE_DIR
)

# Define common include directories
set(COMMON_INCLUDE_DIRS ${EXECUTORCH_SOURCE_INCLUDE_DIR})

# The backend resolves OpenVINO C API symbols via dlopen/dlsym at runtime, so
# there is no build-time dependency on the OpenVINO SDK.

# Shared when the wheel build asks for it, static otherwise.
if(EXECUTORCH_BUILD_SHARED)
  set(_openvino_backend_library_type SHARED)
else()
  set(_openvino_backend_library_type STATIC)
endif()
add_library(openvino_backend ${_openvino_backend_library_type})

# Enable exceptions and RTTI for OpenVINO backend
target_compile_options(openvino_backend PRIVATE -frtti -fexceptions)

# Add source files for OpenVINO backend
target_sources(
  openvino_backend
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/runtime/OpenvinoBackend.cpp
)

# Include Executorch directories
target_include_directories(openvino_backend PRIVATE ${COMMON_INCLUDE_DIRS})

# Link ExecuteTorch core and dynamic loading libraries
target_link_libraries(openvino_backend PRIVATE ${CMAKE_DL_LIBS})

if(EXECUTORCH_BUILD_SHARED)
  # Named after what the library provides rather than after the target that
  # produces it, matching the other shipped delegates, so the file reads as
  # libexecutorch_backend_openvino.so.
  set_target_properties(
    openvino_backend PROPERTIES OUTPUT_NAME executorch_backend_openvino
  )
  executorch_target_soname_policy(openvino_backend)
  # The shared runtime, never also the static core: linking both would compile
  # the backend registry into this library as well and give the process two of
  # them.
  target_link_libraries(openvino_backend PUBLIC executorch_shared)
  # Ships beside the runtime in the wheel's lib/ directory.
  executorch_target_shipped_runtime_path(openvino_backend)
else()
  target_link_libraries(openvino_backend PUBLIC executorch_core)
endif()

executorch_target_link_options_shared_lib(openvino_backend)

# Install OpenVINO backend library to the lib directory
install(
  TARGETS openvino_backend
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
