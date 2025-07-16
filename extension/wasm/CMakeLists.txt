
cmake_minimum_required(VERSION 3.24) # 3.24 is required for WHOLE_ARCHIVE

project(executorch_wasm)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT EMSCRIPTEN)
  message(FATAL_ERROR "Emscripten is required to build this target")
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
set(_common_compile_options -Wno-deprecated-declarations -fPIC)
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

set(link_libraries)
list(
  APPEND
  link_libraries
  embind
  executorch_core
  extension_data_loader
  portable_ops_lib
  extension_module_static
  extension_tensor
  extension_runner_util
)

add_executable(executorch_wasm wasm_bindings.cpp)

target_compile_options(executorch_wasm PUBLIC ${_common_compile_options})
target_include_directories(executorch_wasm PUBLIC ${_common_include_directories})
target_link_libraries(executorch_wasm PUBLIC ${link_libraries})
target_link_options(executorch_wasm PUBLIC -sALLOW_MEMORY_GROWTH=1 --embed-file=${CMAKE_CURRENT_SOURCE_DIR}/model.pte@model.pte)

add_custom_target(executorch_wasm_test ALL
  COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/test_module.js ${CMAKE_CURRENT_BINARY_DIR}/test_module.js
  DEPENDS executorch_wasm
  COMMENT "Copying test_module.js to build output directory"
)
