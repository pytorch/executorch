cmake_minimum_required(VERSION 3.10)
project(pico)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../)


set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/arm-cortex-m0plus-toolchain.cmake)

# Only include here, NOT in arm-cortex-m0plus-toolchain.cmake
include(${CMAKE_SOURCE_DIR}/tools/cmake/Codegen.cmake)

target_sources(executorch_core PRIVATE ${CMAKE_CURRENT_LIST_DIR}/syscall_stubs.c)
gen_selected_ops(LIB_NAME "portable_ops_lib" ROOT_OPS "aten::add.out")
