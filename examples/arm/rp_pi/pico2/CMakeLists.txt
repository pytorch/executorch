cmake_minimum_required(VERSION 3.13)
include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)
project(executorch_pico C CXX ASM)
pico_sdk_init()

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Replace xxx_pte.c with your model's pte file
add_executable(executorch_pico main.cpp simple_addmodule_pte.c)
pico_enable_stdio_usb(executorch_pico 1)
pico_enable_stdio_uart(executorch_pico 0)

target_include_directories(executorch_pico PRIVATE
  ${EXECUTORCH_ROOT}/executorch/runtime/core/portable_type/c10
)
add_compile_definitions(C10_USING_CUSTOM_GENERATED_MACROS)

# Set correct flags for Pico (Cortex-M0+)
target_compile_options(executorch_pico PRIVATE
    -mcpu=cortex-m0plus
    -mfloat-abi=soft
    -mthumb
)

set(BAREMETAL_BUILD_DIR ${EXECUTORCH_ROOT}/executorch/baremetal_build/)
# Link Executorch and Pico libraries
target_link_libraries(executorch_pico
    PRIVATE
    ${BAREMETAL_BUILD_DIR}/libexecutorch.a
    ${BAREMETAL_BUILD_DIR}/libexecutorch_core.a
    -Wl,--whole-archive
    ${BAREMETAL_BUILD_DIR}/kernels/portable/libportable_ops_lib.a
    -Wl,--no-whole-archive
    ${BAREMETAL_BUILD_DIR}/kernels/portable/libportable_kernels.a
    pico_stdlib
    pico_stdio_usb
)

# Include Executorch and third-party headers
target_include_directories(executorch_pico PRIVATE
    ${EXECUTORCH_ROOT}
    ${EXECUTORCH_ROOT}/executorch/third-party/
    # Add other include paths as needed
)

#set(CMAKE_C_FLAGS "-Os -ffunction-sections -fdata-sections")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--gc-sections")

pico_add_extra_outputs(executorch_pico)
