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

# Define common include directories
set(COMMON_INCLUDE_DIRS ${EXECUTORCH_ROOT}/..)

# Include utility CMake scripts from ExecuteTorch
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

# Find OpenVINO libraries
find_package(OpenVINO REQUIRED)

# Define OpenVINO backend as a static library
add_library(openvino_backend STATIC .)

# Enable exceptions and RTTI for OpenVINO backend
target_compile_options(openvino_backend PRIVATE -frtti -fexceptions)

# Include Executorch directories
target_include_directories(openvino_backend PRIVATE ${COMMON_INCLUDE_DIRS})

# Link OpenVINO and ExecuteTorch core libraries
target_link_libraries(
  openvino_backend PRIVATE openvino::runtime executorch_core
)

# Add source files for OpenVINO backend
target_sources(
  openvino_backend
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/runtime/OpenvinoBackend.cpp
)

executorch_target_link_options_shared_lib(openvino_backend)

# Install OpenVINO backend library to the lib directory
install(
  TARGETS openvino_backend
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
