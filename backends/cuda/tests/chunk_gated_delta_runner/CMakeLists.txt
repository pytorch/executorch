cmake_minimum_required(VERSION 3.24)
project(chunk_gated_delta_runner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

set(_common_include_directories ${EXECUTORCH_ROOT}/..)

set(gflags_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../../third-party/gflags)
find_package(gflags REQUIRED)

list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../../..)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
executorch_target_link_options_shared_lib(executorch)

set(link_libraries executorch gflags)

list(APPEND link_libraries optimized_native_cpu_ops_lib cpublas eigen_blas)
executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)

list(
  APPEND
  link_libraries
  extension_module
  extension_data_loader
  extension_tensor
  extension_flat_tensor
  extension_named_data_map
)

if(EXECUTORCH_BUILD_CUDA)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND link_libraries aoti_cuda_backend)
  if(NOT MSVC)
    executorch_target_link_options_shared_lib(aoti_cuda_backend)
  endif()
endif()

add_executable(chunk_gated_delta_runner main.cpp)
target_include_directories(
  chunk_gated_delta_runner PUBLIC ${_common_include_directories}
)
target_link_libraries(chunk_gated_delta_runner PUBLIC ${link_libraries})

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(chunk_gated_delta_runner)
  if(NOT APPLE AND NOT MSVC)
    target_link_options(chunk_gated_delta_runner PRIVATE "LINKER:-s")
  endif()
endif()
