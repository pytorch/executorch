# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure compile_commands are generated
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Define common include directories
set(COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

# Include common directories before others to ensure proper order
include_directories(BEFORE ${COMMON_INCLUDE_DIRS})

# Set up EXECUTORCH_ROOT if not already set
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

# Include utility cmake script from the executorch repository
include(${EXECUTORCH_ROOT}/build/Utils.cmake)

# Update common include directory for ExecuteTorch
set(COMMON_INCLUDE_DIRS ${EXECUTORCH_ROOT}/..)

# Set OpenVINO directory and include directories from environment variable
set(OPENVINO_DIR "$ENV{INTEL_OPENVINO_DIR}")
if(NOT OPENVINO_DIR)
    message(FATAL_ERROR "INTEL_OPENVINO_DIR environment variable is not set.")
endif()

set(OPENVINO_INCLUDE_DIRS
    ${OPENVINO_DIR}/deployment_tools/inference_engine/include
    ${OPENVINO_DIR}/runtime/include
)

# Define OpenVINO library path
set(OPENVINO_LIB_PATH ${OPENVINO_DIR}/runtime/lib/intel64)

# Define OpenVINO libraries
set(OPENVINO_LIB ${OPENVINO_LIB_PATH}/libopenvino.so)

# Add the OpenVINO backend library as a shared library
add_library(openvino_backend SHARED)

# Enable exceptions and RTTI for OpenVINO backend
target_compile_options(openvino_backend PRIVATE "-frtti" "-fexceptions")

# Include directories for ExecuteTorch and OpenVINO
target_include_directories(
  openvino_backend PUBLIC
  ${COMMON_INCLUDE_DIRS}
  ${OPENVINO_INCLUDE_DIRS}
)

# Link OpenVINO libraries and executorch core to the backend
target_link_libraries(openvino_backend PRIVATE
  ${OPENVINO_LIB}
  executorch_core
)

# Add source files to the OpenVINO backend library
target_sources(openvino_backend PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/runtime/OpenvinoBackend.cpp
)

# Set additional link options for shared library
target_link_options(openvino_backend PRIVATE -Wl,-rpath=${OPENVINO_LIB_PATH})

# Install the OpenVINO backend library to the lib directory
install(TARGETS openvino_backend DESTINATION lib)

