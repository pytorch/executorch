cmake_minimum_required(VERSION 3.19)
project(openvino_backend)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/build/Utils.cmake)

set(_common_include_directories ${EXECUTORCH_ROOT}/..)

# Set openvino directory from environment
set(OPENVINO_DIR "$ENV{INTEL_OPENVINO_DIR}")
set(OPENVINO_INCLUDE_DIRS ${OPENVINO_DIR}/deployment_tools/inference_engine/include ${OPENVINO_DIR}/runtime/include)
message("${OPENVINO_DIR}/runtime/include/openvino")

# Define the source files for the OpenVINO backend
set(_openvino_backend_sources backends/openvino/runtime/OpenvinoBackend.cpp)

list(TRANSFORM _openvino_backend_sources PREPEND "${EXECUTORCH_ROOT}/")

# Add the OpenVINO backend library
add_library(openvino_backend STATIC ${_openvino_backend_sources})

# Include directories for ExecuteTorch and OpenVINO
target_include_directories(
  openvino_backend PUBLIC ${_common_include_directories}
)

target_include_directories(
  openvino_backend PUBLIC ${OPENVINO_INCLUDE_DIRS}
)

set(OPENVINO_LIB_PATH ${OPENVINO_DIR}/runtime/lib/intel64)
set(OPENVINO_LIBS
    ${OPENVINO_LIB_PATH}/libopenvino.so
    ${OPENVINO_LIB_PATH}/libopenvino_ir_frontend.so.2430
    ${OPENVINO_LIB_PATH}/libopenvino_c.so
    ${OPENVINO_LIB_PATH}/libopenvino_intel_cpu_plugin.so
    ${OPENVINO_LIB_PATH}/libopenvino_intel_gpu_plugin.so
    ${OPENVINO_LIB_PATH}/libopenvino_auto_plugin.so
)

# Link the OpenVINO library to the backend
target_link_libraries(openvino_backend PRIVATE OPENVINO_LIBS)

