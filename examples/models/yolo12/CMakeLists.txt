cmake_minimum_required(VERSION 3.5)

project(Yolo12DetectionDemo VERSION 0.1)

option(USE_OPENVINO_BACKEND "Build the tutorial with the OPENVINO backend" OFF)
option(USE_XNNPACK_BACKEND "Build the tutorial with the XNNPACK backend" OFF)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# !OpenCV

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(TORCH_ROOT ${EXECUTORCH_ROOT}/third-party/pytorch)

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

# find `executorch` libraries Same as for gflags
find_package(executorch CONFIG REQUIRED PATHS ${EXECUTORCH_ROOT}/cmake-out)
executorch_target_link_options_shared_lib(executorch)

add_subdirectory(${EXECUTORCH_ROOT}/third-party/gflags gflags)
set(link_libraries gflags)
list(APPEND link_libraries portable_ops_lib portable_kernels)
executorch_target_link_options_shared_lib(portable_ops_lib)

if(USE_XNNPACK_BACKEND)
  set(xnnpack_backend_libs xnnpack_backend XNNPACK xnnpack-microkernels-prod)
  list(APPEND link_libraries ${xnnpack_backend_libs})
  executorch_target_link_options_shared_lib(xnnpack_backend)
endif()

if(USE_OPENVINO_BACKEND)
  find_package(OpenVINO REQUIRED)
  list(APPEND link_libraries openvino_backend)
  executorch_target_link_options_shared_lib(openvino_backend)
endif()

list(APPEND link_libraries extension_threadpool pthreadpool)
list(APPEND _common_include_directories
     ${XNNPACK_ROOT}/third-party/pthreadpool/include
)

set(PROJECT_SOURCES
    main.cpp
    inference.h
    ${EXECUTORCH_ROOT}/extension/data_loader/file_data_loader.cpp
    ${EXECUTORCH_ROOT}/extension/evalue_util/print_evalue.cpp
    ${EXECUTORCH_ROOT}/extension/runner_util/inputs.cpp
    ${EXECUTORCH_ROOT}/extension/runner_util/inputs_portable.cpp
)

add_executable(Yolo12DetectionDemo ${PROJECT_SOURCES})
target_link_libraries(
  Yolo12DetectionDemo PUBLIC ${link_libraries} ${OpenCV_LIBS} executorch_core
                             extension_module extension_tensor
)

find_package(Threads REQUIRED)
target_link_libraries(Yolo12DetectionDemo PRIVATE Threads::Threads)
target_include_directories(
  Yolo12DetectionDemo PUBLIC ${_common_include_directories}
)
