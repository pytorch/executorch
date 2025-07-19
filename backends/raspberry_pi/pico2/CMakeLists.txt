cmake_minimum_required(VERSION 3.13)
include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)
project(executorch_pico C CXX ASM)
pico_sdk_init()

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(executorch_pico main.cpp simple_addmodule_pte.c)
pico_enable_stdio_usb(executorch_pico 1)
pico_enable_stdio_uart(executorch_pico 0)

# Set correct flags for Pico (Cortex-M0+)
target_compile_options(executorch_pico PRIVATE
    -mcpu=cortex-m0plus
    -mfloat-abi=soft
    -mthumb
)

# Link Executorch and Pico libraries
target_link_libraries(executorch_pico
    PRIVATE
    ${EXECUTORCH_ROOT}/executorch/build/libexecutorch.a
    ${EXECUTORCH_ROOT}/executorch/build/libexecutorch_core.a
    -Wl,--whole-archive
    ${EXECUTORCH_ROOT}/executorch/build/kernels/portable/libportable_ops_lib.a
    -Wl,--no-whole-archive
    ${EXECUTORCH_ROOT}/executorch/build/kernels/portable/libportable_kernels.a
    pico_stdlib
    pico_stdio_usb
)

# Include Executorch and third-party headers
target_include_directories(executorch_pico PRIVATE
    ${EXECUTORCH_ROOT}
    ${EXECUTORCH_ROOT}/executorch/third-party/
    # Add other include paths as needed
)

pico_add_extra_outputs(executorch_pico)
