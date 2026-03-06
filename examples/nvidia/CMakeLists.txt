cmake_minimum_required(VERSION 3.19)

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

if(EXECUTORCH_BUILD_TENSORRT)
  add_subdirectory(tensorrt)
endif()
